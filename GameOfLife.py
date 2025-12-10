import glfw
import moderngl
import numpy as np
import math
import time

# ------------------------------------------------------------------
# Configuration: logical simulation/grid size (NEVER changes)
# ------------------------------------------------------------------
GRID_WIDTH = 512  # logical simulation width (texture width)
GRID_HEIGHT = 512  # logical simulation height (texture height)


# -----------------------------
# Initialize GLFW and fullscreen window
# -----------------------------
def glfw_window_init(fullscreen=True):
    if not glfw.init():
        raise Exception("GLFW failed to initialize")

    monitor = glfw.get_primary_monitor()
    mode = glfw.get_video_mode(monitor)
    if fullscreen:
        width, height = mode.size.width, mode.size.height
        window = glfw.create_window(width, height, "GPU Conway's Game of Life", monitor, None)
    else:
        width, height = 1280, 720
        glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
        window = glfw.create_window(width, height, "GPU Conway's Game of Life", None, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window failed to create")

    glfw.make_context_current(window)
    return window, width, height


# -----------------------------
# Create glfw window helper (used when toggling)
# -----------------------------
def create_window(fullscreen=False, width=1280, height=720, title="GPU Conway's Game of Life"):
    monitor = glfw.get_primary_monitor() if fullscreen else None
    if fullscreen:
        mode = glfw.get_video_mode(monitor)
        width, height = mode.size.width, mode.size.height

    glfw.window_hint(glfw.RESIZABLE, glfw.TRUE)
    window = glfw.create_window(width, height, title, monitor, None)

    if not window:
        glfw.terminate()
        raise Exception("GLFW window failed to create")

    glfw.make_context_current(window)
    return window


# -----------------------------
# ModernGL context
# -----------------------------
def modernGL_context_init(width, height):
    ctx = moderngl.create_context()
    ctx.viewport = (0, 0, width, height)
    return ctx


# -----------------------------
# Create initial grid state (fixed logical texture size)
# -----------------------------
def create_initial_grid(ctx, grid_width, grid_height):
    # Create a uint8 grayscale grid, seed the center
    grid = np.zeros((grid_height, grid_width), dtype=np.uint8)

    seed_size = 40
    half = seed_size // 2
    cx = grid_width // 2
    cy = grid_height // 2
    grid[cy-half:cy+half, cx-half:cx+half] = np.random.randint(0, 2, (seed_size, seed_size), dtype=np.uint8) * 255

    # replicate to RGBA channels and pass raw bytes
    rgba = np.repeat(grid[:, :, None], 4, axis=2).astype(np.uint8)
    state_tex = ctx.texture((grid_width, grid_height), components=4, data=rgba.tobytes())
    next_tex = ctx.texture((grid_width, grid_height), components=4, data=rgba.tobytes())

    for tex in (state_tex, next_tex):
        tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        tex.repeat_x = True
        tex.repeat_y = True

    return state_tex, next_tex


# -----------------------------
# Compute shader (created once; size is the fixed grid size)
# -----------------------------
def create_compute_shader(ctx, grid_width, grid_height):
    compute_shader_source = """
    #version 430
    layout(local_size_x = 16, local_size_y = 16) in;

    layout(binding = 0) uniform sampler2D state_tex;
    layout(binding = 1, rgba8) writeonly uniform image2D next_tex;
    uniform ivec2 size;

    void main() {
        ivec2 p = ivec2(gl_GlobalInvocationID.xy);
        if (p.x >= size.x || p.y >= size.y) return;

        int alive = 0;
        for (int dx=-1; dx<=1; dx++) {
            for (int dy=-1; dy<=1; dy++) {
                if (dx==0 && dy==0) continue;
                ivec2 n = (p + ivec2(dx, dy) + size) % size;
                alive += texelFetch(state_tex, n, 0).r > 0.0 ? 1 : 0;
            }
        }

        int current = texelFetch(state_tex, p, 0).r > 0.0 ? 1 : 0;

        bool birth   = (alive == 3);
        bool survive = (current == 1);
        int next = (birth || survive) ? 255 : 0;

        imageStore(next_tex, p, vec4(next, next, next, 255));
    }
    """
    cs = ctx.compute_shader(compute_shader_source)
    cs['size'] = (grid_width, grid_height)
    return cs


# -----------------------------
# Fullscreen quad for rendering
# returns (vao, program) so we can set u_scale
# -----------------------------
def create_fullscreen_quad(ctx):
    vertices = np.array([
        -1.0, -1.0, 0.0, 0.0,
         1.0, -1.0, 1.0, 0.0,
        -1.0,  1.0, 0.0, 1.0,
         1.0,  1.0, 1.0, 1.0,
    ], dtype='f4')

    vbo = ctx.buffer(vertices.tobytes())
    prog = ctx.program(
        vertex_shader="""
        #version 330
        in vec2 in_vert;
        in vec2 in_uv;
        out vec2 v_uv;
        uniform vec2 u_scale; // scale the quad to preserve texture aspect
        void main() {
            vec2 pos = in_vert * u_scale;
            gl_Position = vec4(pos, 0.0, 1.0);
            v_uv = in_uv;
        }
        """,
        fragment_shader="""
        #version 330
        uniform sampler2D tex;
        in vec2 v_uv;
        out vec4 f_color;
        void main() {
            vec4 c = texture(tex, v_uv);
            f_color = vec4(c.r, 0.2, 0.2, 1.0);
        }
        """
    )
    vao = ctx.simple_vertex_array(prog, vbo, 'in_vert', 'in_uv')
    return vao, prog


# -----------------------------
# Helper: compute scale to draw texture into window while preserving aspect
# returns (scaleX, scaleY) to multiply the full [-1,1] quad
# -----------------------------
def compute_aspect_scale(grid_w, grid_h, win_w, win_h):
    tex_aspect = grid_w / grid_h
    win_aspect = win_w / win_h if win_h != 0 else 1.0
    if win_aspect >= tex_aspect:
        # window is wider → full height, shrink X
        scale_x = tex_aspect / win_aspect
        scale_y = 1.0
    else:
        # window is taller → full width, shrink Y
        scale_x = 1.0
        scale_y = win_aspect / tex_aspect
    return scale_x, scale_y


# -----------------------------
# Main loop
# -----------------------------
def main_loop(window, ctx, state_tex, next_tex, compute_shader, vao, prog, grid_width, grid_height, start_fb_width, start_fb_height):
    # grid_width/grid_height are fixed (texture size)
    fb_w, fb_h = start_fb_width, start_fb_height

    fullscreen = True
    f11_was_pressed = False

    while not glfw.window_should_close(window):
        glfw.poll_events()

        # Safe F11 toggle (debounced)
        if glfw.get_key(window, glfw.KEY_F11) == glfw.PRESS:
            if not f11_was_pressed:
                fullscreen = not fullscreen
                monitor = glfw.get_primary_monitor()
                if fullscreen:
                    mode = glfw.get_video_mode(monitor)
                    glfw.set_window_monitor(window, monitor, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
                else:
                    # set to windowed 1280x720 positioned at (100,100)
                    glfw.set_window_monitor(window, None, 100, 100, 1280, 720, 0)
                # After mode change, get framebuffer size and reapply viewport below
                f11_was_pressed = True
        else:
            f11_was_pressed = False

        # Get framebuffer size every frame (may change on resize or fullscreen toggle)
        new_fb_w, new_fb_h = glfw.get_framebuffer_size(window)
        if new_fb_w != fb_w or new_fb_h != fb_h:
            fb_w, fb_h = new_fb_w, new_fb_h
            # Update viewport (note: texture size doesn't change)
            ctx.viewport = (0, 0, fb_w, fb_h)

        # --------------------------
        # Run compute shader (always against fixed logical grid size)
        # --------------------------
        state_tex.use(location=0)
        next_tex.bind_to_image(1, read=False, write=True)

        groups_x = math.ceil(grid_width / 16)
        groups_y = math.ceil(grid_height / 16)
        compute_shader.run(groups_x, groups_y, 1)

        # swap textures (logical ping-pong)
        state_tex, next_tex = next_tex, state_tex

        # --------------------------
        # Render the logical texture into the window without reallocating it
        # (preserve aspect; compute u_scale so quad is letterboxed if needed)
        # --------------------------
        scale_x, scale_y = compute_aspect_scale(grid_width, grid_height, fb_w, fb_h)
        prog['u_scale'].value = (scale_x, scale_y)

        ctx.clear(0.0, 0.0, 0.0)
        state_tex.use(location=0)
        vao.render(moderngl.TRIANGLE_STRIP)

        glfw.swap_buffers(window)
        # sleep to throttle; tune as desired
        time.sleep(0.01)

    glfw.terminate()


def play():
    # Start fullscreen window (doesn't matter — texture grid size is fixed below)
    window, fb_w, fb_h = glfw_window_init(fullscreen=True)
    ctx = modernGL_context_init(fb_w, fb_h)

    # create rendering VAO/prog
    vao, prog = create_fullscreen_quad(ctx)

    # create compute shader and the fixed-size textures (GRID_WIDTH x GRID_HEIGHT)
    compute_cs = create_compute_shader(ctx, GRID_WIDTH, GRID_HEIGHT)
    state_tex, next_tex = create_initial_grid(ctx, GRID_WIDTH, GRID_HEIGHT)

    # run main loop; note we pass both logical grid size and framebuffer size
    main_loop(window, ctx, state_tex, next_tex, compute_cs, vao, prog, GRID_WIDTH, GRID_HEIGHT, fb_w, fb_h)


if __name__ == "__main__":
    play()
    