#!/usr/bin/env python3
"""
X11 显示 SE 诊断结果 — 窗口保持打开直到用户手动关闭。
用法:
  python scripts/display_se_results.py output/se_pipeline/72h
"""
import sys
import os
from pathlib import Path
from tkinter import Tk, Label, Frame, BOTH, YES, BOTTOM, LEFT
from PIL import Image, ImageTk

os.environ.setdefault("DISPLAY", "localhost:10.0")


def show_images(forcing_png: str, solution_png: str) -> None:
    root = Tk()
    root.title("SE 诊断结果 — 72h | 关闭窗口退出")
    root.configure(bg="#2b2b2b")

    # 状态栏提示
    status = Label(
        root,
        text="SE Diagnostic Results @ 72h  |  左: 强迫项 (A–F)  |  右: 解场 (ψ, U_se, W_se, Vt)  |  关闭窗口即可退出",
        bg="#1a1a1a", fg="#aaaaaa", font=("Helvetica", 11), pady=6,
    )
    status.pack(side=BOTTOM, fill=BOTH)

    # 主体框架
    frame = Frame(root, bg="#2b2b2b")
    frame.pack(fill=BOTH, expand=YES, padx=4, pady=4)

    # --- 左侧：强迫项 ---
    img1 = Image.open(forcing_png)
    photo1 = ImageTk.PhotoImage(img1)
    lbl1 = Label(frame, image=photo1, bg="#2b2b2b")
    lbl1.image = photo1  # 保持引用防止 GC
    lbl1.pack(side=LEFT, fill=BOTH, expand=YES, padx=2)

    # --- 右侧：解场 ---
    img2 = Image.open(solution_png)
    photo2 = ImageTk.PhotoImage(img2)
    lbl2 = Label(frame, image=photo2, bg="#2b2b2b")
    lbl2.image = photo2
    lbl2.pack(side=LEFT, fill=BOTH, expand=YES, padx=2)

    # 设置窗口大小（两个图并排，尽量铺满但不超出屏幕）
    screen_w = root.winfo_screenwidth()
    screen_h = root.winfo_screenheight()
    win_w = min(screen_w - 60, img1.width + img2.width + 30)
    win_h = min(screen_h - 80, max(img1.height, img2.height) + 50)
    root.geometry(f"{win_w}x{win_h}+20+20")

    print("[INFO] 窗口已打开，请在你的 Mac XQuartz 中查看。")
    print("[INFO] 关闭窗口即可退出。")
    root.mainloop()
    print("[INFO] 窗口已关闭。")


def main() -> None:
    if len(sys.argv) >= 2:
        out_dir = Path(sys.argv[1])
    else:
        out_dir = Path("output/se_pipeline/72h")

    forcing_png = out_dir / "se_forcing_terms.png"
    solution_png = out_dir / "se_solution_fields.png"

    for p in [forcing_png, solution_png]:
        if not p.exists():
            print(f"[ERROR] 文件不存在: {p}")
            sys.exit(1)

    print(f"[INFO] 加载图像: {forcing_png}")
    print(f"[INFO] 加载图像: {solution_png}")
    show_images(str(forcing_png), str(solution_png))


if __name__ == "__main__":
    main()
