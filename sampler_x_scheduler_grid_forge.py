import math
import gradio as gr
import modules.scripts as scripts
from modules.sd_samplers import samplers
from modules.sd_schedulers import schedulers
from modules.processing import process_images, Processed
from modules import shared
from modules.shared import state
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
import logging
import sys
import time
import re
import gc

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)

BASE_DIR = Path(scripts.basedir())
OUTPUT_ROOT = BASE_DIR / "outputs" / "sampler_scheduler_grid_forge"
output_dir = OUTPUT_ROOT
output_dir.mkdir(parents=True, exist_ok=True)
cells_dir = output_dir / "cells"
cells_dir.mkdir(exist_ok=True)

LIMIT = 16383
label_gap = 6
auto_downscale = True


def sanitize_filename(s):
    return re.sub(r'[<>:"/\\|?*\n\r]+', '_', s).strip()[:200]


def get_next_grid_index(output_root: Path, prefix: str = "grid_") -> str:

    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)

    pattern = re.compile(
        rf"^{re.escape(prefix)}{timestamp}_(\d+)\.(?:png|webp)$"
    )

    existing = [
        f for f in output_root.iterdir()
        if f.is_file() and pattern.match(f.name)
    ]

    indices = [int(pattern.match(f.name).group(1)) for f in existing]
    next_idx = max(indices, default=0) + 1

    return f"{prefix}{timestamp}_{next_idx:03d}"


def generate_or_warn(p, sampler, scheduler, font_path):
    try:
        apply_params(p, sampler, scheduler)

        res = process_images(p)
        img = res.images[0] if res and getattr(res, "images", None) else None
        if img is None:
            raise ValueError()

    except Exception as e:
        logger.warning(
            f"⚠️ Generation failed for Sampler = '{sampler}', Scheduler = '{scheduler}': {e}")
        img = create_fallback_image(
            p.width, p.height, sampler, scheduler, font_path)

    return img


def apply_params(p, sampler_label, scheduler_label):
    sampler_label = sampler_label.strip()
    sampler_index = next((i for i, s in enumerate(
        samplers) if s.name == sampler_label), None)
    if sampler_index is None:
        raise ValueError(f"Sampler '{sampler_label}' not found")

    p.sampler_index = sampler_index
    p.sampler_name = samplers[sampler_index].name
    p.scheduler = scheduler_label.strip()

    logger.info(
        f"🧪 Applied Sampler = '{sampler_label}' → index {sampler_index}")
    logger.info(f"🧪 Applied Scheduler = '{scheduler_label.strip()}'")


def create_fallback_image(width, height, sampler, scheduler, font_path):
    img = Image.new("RGB", (width, height), (255, 230, 230))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(
        str(font_path), 38) if font_path else ImageFont.load_default()

    msg = f"⚠️ Pair failed\n{sampler} × {scheduler}"
    lines = msg.split("\n")

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    total_h = len(lines) * line_h
    y = (height - total_h) // 2

    for line in lines:
        text_w = font.getbbox(line)[2]
        x = (width - text_w) // 2
        draw.text((x, y), line, font=font, fill=(150, 0, 0))
        y += line_h

    return img


def safe_processed(*args):
    processed = Processed(*args)
    processed.info = str(processed.info or "")
    processed.comments = str(processed.comments or "")
    return processed


def wrap_text(text, font, max_width):
    words = text.split()
    lines, cur = [], ""
    for word in words:
        test = f"{cur} {word}".strip()
        w = font.getbbox(test)[2]
        if w <= max_width - 10:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def wrap_text_to_fit(draw, text, font_path, max_width, initial_size=42, min_size=18):
    for size in range(initial_size, min_size - 1, -1):
        font = ImageFont.truetype(
            str(font_path), size) if font_path else ImageFont.load_default()
        lines = wrap_text(text, font, max_width)
        if all(font.getbbox(line)[2] <= max_width * 0.98 for line in lines) and len(lines) <= 3:
            return lines, font
    font = ImageFont.truetype(
        str(font_path), min_size) if font_path else ImageFont.load_default()
    return wrap_text(text, font, max_width), font


def create_batch_grid(images, width=832, height=1216, padding=10, bg_color=(255, 255, 255)):
    if not images:
        return Image.new("RGB", (width, height), bg_color)

    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)

    cell_w = images[0].width
    cell_h = images[0].height

    grid_w = cols * cell_w + (cols - 1) * padding
    grid_h = rows * cell_h + (rows - 1) * padding

    total_w = grid_w + 2 * padding
    total_h = grid_h + 2 * padding

    grid = Image.new("RGB", (total_w, total_h), bg_color)

    for i, img in enumerate(images):
        r, c = divmod(i, cols)
        x = padding + c * (cell_w + padding)
        y = padding + r * (cell_h + padding)
        grid.paste(img, (x, y))

    return grid


def annotate_batch_image(img, sampler, scheduler, font_path=None):
    label_text = f"{sampler.strip()} × {scheduler.strip()}"
    dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    lines, font = wrap_text_to_fit(
        dummy_draw, label_text, font_path, img.width)

    ascent, descent = font.getmetrics()
    line_h = ascent + descent
    total_text_h = line_h * len(lines)

    top_margin = label_gap
    gap = label_gap

    final_h = top_margin + total_text_h + gap + img.height + label_gap
    final_w = img.width

    out = Image.new("RGB", (final_w, final_h), (255, 255, 255))
    out.paste(img, (0, top_margin + total_text_h + gap))

    draw = ImageDraw.Draw(out)
    y_text = top_margin

    for line in lines:
        text_w = font.getbbox(line)[2]
        x_text = (final_w - text_w) // 2
        draw.text((x_text, y_text), line, font=font, fill=(0, 0, 0))
        y_text += line_h

    return out


class Script(scripts.Script):
    def title(self):
        return "🔬 Sampler × Scheduler Grid (Forge)"

    def show(self, is_img2img):
        return not is_img2img

    def ui(self, is_img2img):
        sampler_list = [s.name for s in samplers if hasattr(s, "name")]
        scheduler_list = [s.label for s in schedulers if hasattr(s, "label")]

        mode_selector = gr.Radio(
            ["XY Grid", "Batch Grid"], value="XY Grid", label="🏁 Grid Mode")

        stop_btn = gr.Button("🛑 Stop Grid Generation")
        stop_btn.click(fn=lambda: shared.state.interrupt(),
                       inputs=[], outputs=[])

        xy_group = gr.Group(visible=True)
        with xy_group:
            with gr.Row():
                with gr.Column():
                    xy_samplers = gr.Dropdown(
                        choices=sampler_list, multiselect=True, label="🗳️ Sampler(s)")
                    select_all_samplers_btn = gr.Button("✅ Select All")
                    clear_all_samplers_btn = gr.Button("🧹 Clear All")
                with gr.Column():
                    xy_schedulers = gr.Dropdown(
                        choices=scheduler_list, multiselect=True, label="📆 Scheduler(s)")
                    select_all_schedulers_btn = gr.Button("✅ Select All")
                    clear_all_schedulers_btn = gr.Button("🧹 Clear All")
            sampler_axis = gr.Radio(
                ["Axis X", "Axis Y"], value="Axis X", label="🧭 Place Sampler on")

        select_all_samplers_btn.click(lambda: gr.update(
            value=sampler_list), inputs=[], outputs=[xy_samplers])
        clear_all_samplers_btn.click(lambda: gr.update(
            value=[]), inputs=[], outputs=[xy_samplers])
        select_all_schedulers_btn.click(lambda: gr.update(
            value=scheduler_list), inputs=[], outputs=[xy_schedulers])
        clear_all_schedulers_btn.click(lambda: gr.update(
            value=[]), inputs=[], outputs=[xy_schedulers])

        batch_group = gr.Group(visible=False)
        with batch_group:
            with gr.Row():
                dropdown_sampler = gr.Dropdown(
                    choices=sampler_list, label="🗳️ Sampler(s)")
                dropdown_scheduler = gr.Dropdown(
                    choices=scheduler_list, label="📆 Scheduler(s)")
            add_pair_btn = gr.Button("➕ Add Pair")
            clear_pairs_btn = gr.Button("🧹 Clear All Pairs")
            pair_list = gr.Textbox(
                label="🔗 Added Pairs", placeholder="Sampler, Scheduler per line", lines=6)
            pair_count = gr.Textbox(label="🧮 Total Pairs", interactive=False)
            pair_state = gr.State([])

            def parse_pairs(txt):
                lines = [line.strip()
                         for line in txt.splitlines() if "," in line]
                return list(dict.fromkeys(lines))

            pair_list.change(lambda txt: (parse_pairs(txt), str(len(parse_pairs(txt)))), inputs=[
                             pair_list], outputs=[pair_state, pair_count])
            add_pair_btn.click(lambda s, sch, cur: cur + [f"{s},{sch}"] if s and sch and f"{s},{sch}" not in cur else cur,
                               inputs=[dropdown_sampler, dropdown_scheduler, pair_state], outputs=[pair_state])
            pair_state.change(lambda st: ("\n".join(st), str(len(st))), inputs=[
                              pair_state], outputs=[pair_list, pair_count])
            clear_pairs_btn.click(lambda: [], [], [pair_state])

        pos_prompt = gr.Textbox(label="✅ Positive Prompt",
                                placeholder="What to include", lines=3)
        neg_prompt = gr.Textbox(label="⛔ Negative Prompt",
                                placeholder="What to avoid", lines=2)
        seed = gr.Textbox(label="🎲 Seed (optional)",
                          placeholder="Leave blank for random")
        steps = gr.Slider(1, 100, value=35, step=1, label="🚀 Steps")
        cfg_scale = gr.Slider(1.0, 30.0, value=5, step=1, label="🎯 CFG Scale")
        width = gr.Slider(256, 2048, value=832, step=1, label="↔️ Width")
        height = gr.Slider(256, 2048, value=1216, step=1, label="↕️ Height")
        padding = gr.Slider(0, 200, value=20, step=1, label="📏 Padding (px)")
        save_formats = gr.CheckboxGroup(choices=["WEBP", "PNG"], value=[
                                        "WEBP"], label="💾 Save As")
        show_labels = gr.Checkbox(label="📝 Add Labels", value=True)
        save_cells = gr.Checkbox(
            label="💾 Save each cell individually", value=False)

        mode_selector.change(lambda m: {
            xy_group: gr.update(visible=m == "XY Grid"),
            batch_group: gr.update(visible=m == "Batch Grid")
        }, inputs=[mode_selector], outputs=[xy_group, batch_group])

        return [
            mode_selector,
            xy_samplers, xy_schedulers, sampler_axis,
            dropdown_sampler, dropdown_scheduler,
            pair_list, pair_state, pair_count,
            pos_prompt, neg_prompt, seed, steps, cfg_scale,
            width, height, padding,
            save_formats, show_labels, save_cells
        ]

    def run(self, p, *args):
        (mode, xy_samplers, xy_schedulers, sampler_axis,
         dropdown_sampler, dropdown_scheduler,
         pair_list, pair_state, pair_count,
         pos_prompt, neg_prompt, seed, steps, cfg_scale,
         width, height, padding,
         save_formats, show_labels, save_cells) = args

        if state.interrupted:
            logger.warning("🧹 Resetting interrupted state before start")
            state.interrupted = False

        try:
            sd = int(seed.strip()) if seed.strip(
            ) else random.randint(1, 2**32 - 1)
        except ValueError:
            sd = random.randint(1, 2**32 - 1)

        p.seed = sd
        p.prompt = pos_prompt
        p.negative_prompt = neg_prompt
        p.steps = steps
        p.cfg_scale = cfg_scale
        p.width = int(width)
        p.height = int(height)

        p.extra_generation_params = {}

        font_path = Path(__file__).resolve().parent / "Barlow-SemiBold.ttf"
        if not font_path.exists():
            logger.warning("⚠️ Font not found — using default")
            font_path = None

        if mode == "Batch Grid":
            pairs = [line.strip()
                     for line in pair_list.strip().splitlines() if "," in line]
            if not pairs:
                logger.warning("⚠️ No valid pairs found in 🔗 Added Pairs.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No pairs found", "")

            duplicates = list(
                set([line for line in pairs if pairs.count(line) > 1]))
            if duplicates:
                logger.warning(
                    f"⚠️ Duplicate pair(s) detected: {', '.join(duplicates)} — generation stopped.")
                return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ Duplicate pairs detected", "")

            unique_pairs = list(dict.fromkeys(pairs))
            result_images = []

            for i, line in enumerate(unique_pairs, 1):
                if state.interrupted:
                    print("🛑 Interrupted by user.", flush=True)
                    logger.warning("🛑 Grid generation interrupted by user.")
                    break

                parts = [part.strip() for part in line.split(",", 1)]
                if len(parts) != 2:
                    logger.warning(f"⚠️ Invalid pair format: {line}")
                    continue

                sampler, scheduler = parts
                print(
                    f"🔄[{i}/{len(unique_pairs)}] Sampler = '{sampler}', Scheduler = '{scheduler}'", flush=True)
                logger.info(
                    f"🔄[{i}/{len(unique_pairs)}] Sampler = '{sampler}', Scheduler = '{scheduler}'")

                img = generate_or_warn(
                    p, sampler=sampler, scheduler=scheduler, font_path=font_path)

                if show_labels:
                    img_annotated = annotate_batch_image(
                        img, sampler, scheduler, font_path)
                    result_images.append(img_annotated)
                else:
                    result_images.append(img)

                if save_cells:
                    filename = f"{sanitize_filename(sampler)}__{sanitize_filename(scheduler)}.png"
                    img.save(cells_dir / filename, "PNG")
                    logger.info(f"💾 Batch Cell Saved: {filename}")

            grid = create_batch_grid(result_images, width=p.width, height=p.height,
                                     padding=padding, bg_color=(255, 255, 255))

            grid_idx = get_next_grid_index(output_dir, prefix="batch_grid_")

            for fmt in save_formats:
                ext = fmt.lower()
                out = grid

                if out.width > LIMIT or out.height > LIMIT:
                    if auto_downscale:
                        scale = min(LIMIT / out.width, LIMIT / out.height)
                        out = out.resize(
                            (int(out.width * scale), int(out.height * scale)), Image.LANCZOS)
                        logger.warning(
                            f"🪄 Downscaling Batch grid → {out.width}×{out.height}")
                    else:
                        logger.warning(
                            "⚠️ Image exceeds global LIMIT and auto_downscale disabled — skipping save")
                        continue

                save_kwargs = {"quality": 100} if ext == "webp" else {}
                out.save(
                    output_dir / f"batchgrid_{grid_idx}.{ext}", fmt.upper(), **save_kwargs)
                logger.info(
                    f"💾 Saved batchgrid_{grid_idx}.{ext} ({out.width}×{out.height})")

                if out is not grid:
                    try:
                        out.close()
                    except Exception:
                        pass

            print(
                f"✅ Batch Grid complete: {grid.width}×{grid.height}", flush=True)

            gc.collect()

            return safe_processed(p, [grid], sd, sd, 0.0, pos_prompt, neg_prompt, "✅ Batch Grid complete", "")

        axis_x = "Sampler" if sampler_axis == "Axis X" else "Scheduler"
        axis_y = "Scheduler" if sampler_axis == "Axis X" else "Sampler"

        x_vals = xy_samplers if axis_x == "Sampler" else xy_schedulers
        y_vals = xy_schedulers if axis_y == "Scheduler" else xy_samplers

        if not x_vals:
            print("⚠️ No Sampler(s) selected for Axis X or Y.")
            return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No Sampler(s) selected", "")
        if not y_vals:
            print("⚠️ No Scheduler(s) selected for Axis X or Y.")
            return safe_processed(p, [], sd, sd, 0.0, pos_prompt, neg_prompt, "⚠️ No Scheduler(s) selected", "")

        cells = []
        total = len(x_vals) * len(y_vals)
        i = 1

        for yv in y_vals:
            for xv in x_vals:
                if state.interrupted:
                    print("🛑 Interrupted by user.", flush=True)
                    logger.warning("🛑 Grid generation interrupted by user.")
                    break

                samp = xv if axis_x == "Sampler" else yv
                sched = yv if axis_y == "Scheduler" else xv

                print(
                    f"🔄[{i}/{total}] Sampler = '{samp}', Scheduler = '{sched}'", flush=True)
                logger.info(
                    f"🔄[{i}/{total}] Sampler = '{samp}', Scheduler = '{sched}'")

                img = generate_or_warn(
                    p, sampler=samp, scheduler=sched, font_path=font_path)
                cells.append(img)
                i += 1
            if state.interrupted:
                break

        font_test = ImageFont.truetype(
            str(font_path), 42) if font_path else ImageFont.load_default()
        line_h = sum(font_test.getmetrics())
        dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))

        cols = len(x_vals)
        rows = len(y_vals)

        if show_labels:
            max_label_lines = max(len(wrap_text_to_fit(
                dummy_draw, label, font_path, p.width)[0]) for label in x_vals)
            x_label_h = line_h * max_label_lines + 2 * label_gap

            max_y_label = max(y_vals, key=len) if y_vals else ""
            lines, font = wrap_text_to_fit(
                dummy_draw, max_y_label, font_path, p.height)
            text_height = sum(font.getmetrics()) * len(lines)
            y_label_w = text_height + 2 * label_gap

        else:
            x_label_h = 0
            y_label_w = 0

        grid_w = cols * p.width + (cols - 1) * padding
        grid_h = rows * p.height + (rows - 1) * padding

        full_w = grid_w + (y_label_w if show_labels else 0) + 2 * padding
        full_h = grid_h + (x_label_h if show_labels else 0) + 2 * padding

        grid = Image.new("RGB", (full_w, full_h), (255, 255, 255))
        draw = ImageDraw.Draw(grid)

        if show_labels:
            for ix, label in enumerate(x_vals):
                x = y_label_w + padding + ix * (p.width + padding)
                box_w = p.width
                lines, font = wrap_text_to_fit(draw, label, font_path, box_w)
                line_h = sum(font.getmetrics())
                y_text = padding
                for line in lines:
                    text_w = font.getbbox(line)[2]
                    x_text = x + (box_w - text_w) // 2
                    draw.text((x_text, y_text), line,
                              font=font, fill=(0, 0, 0))
                    y_text += line_h

            for iy, label in enumerate(y_vals):
                y = x_label_h + padding + iy * (p.height + padding)

                dummy_draw = ImageDraw.Draw(Image.new("RGB", (10, 10)))
                lines, font = wrap_text_to_fit(
                    dummy_draw, label, font_path, p.width)

                line_h = sum(font.getmetrics())
                total_text_h = line_h * len(lines)

                x_text = y_label_w - 10 + padding
                y_text = y + (p.height // 2)

                current_y = y_text - (total_text_h // 2)

                for line in lines:
                    text_w = font.getbbox(line)[2]
                    rotated_text = Image.new(
                        "RGBA", (text_w, line_h), (255, 255, 255, 0))
                    rotated_draw = ImageDraw.Draw(rotated_text)
                    rotated_draw.text((0, 0), line, font=font,
                                      fill=(0, 0, 0, 255))
                    rotated_text = rotated_text.rotate(
                        90, expand=True, fillcolor=(255, 255, 255, 0))

                    text_x = y_label_w - rotated_text.width - 10 + padding
                    text_y = current_y + (line_h // 2) - \
                        (rotated_text.height // 2)
                    grid.paste(rotated_text, (text_x, text_y), rotated_text)

                    current_y += line_h

        for idx, img in enumerate(cells):
            r, c = divmod(idx, cols)
            x = (y_label_w if show_labels else 0) + \
                padding + c * (p.width + padding)
            y = (x_label_h if show_labels else 0) + \
                padding + r * (p.height + padding)
            grid.paste(img, (x, y))

            if save_cells:
                filename = f"{sanitize_filename(x_vals[c])}__{sanitize_filename(y_vals[r])}.png"
                img.save(cells_dir / filename, "PNG")
                logger.info(f"💾 XY Cell saved: {filename}")

        grid_idx = get_next_grid_index(output_dir, prefix="xy_grid_")

        for fmt in save_formats:
            ext = fmt.lower()
            out = grid

            if out.width > LIMIT or out.height > LIMIT:
                if auto_downscale:
                    scale = min(LIMIT / out.width, LIMIT / out.height)
                    out = out.resize(
                        (int(out.width * scale), int(out.height * scale)), Image.LANCZOS)
                    logger.warning(
                        f"🪄 Downscaling XY grid → {out.width}×{out.height}")
                else:
                    logger.warning(
                        "⚠️ Image exceeds global LIMIT and auto_downscale disabled — skipping save")
                    continue

            save_kwargs = {"quality": 100} if ext == "webp" else {}
            out.save(output_dir /
                     f"xygrid_{grid_idx}.{ext}", fmt.upper(), **save_kwargs)
            logger.info(
                f"💾 Saved xygrid_{grid_idx}.{ext} ({out.width}×{out.height})")

            if out is not grid:
                try:
                    out.close()
                except Exception:
                    pass

        print(f"✅ XY Grid complete: {grid.width}×{grid.height}", flush=True)

        gc.collect()

        return safe_processed(p, [grid], sd, sd, 0.0, pos_prompt, neg_prompt, "✅ XY Grid complete", "")
