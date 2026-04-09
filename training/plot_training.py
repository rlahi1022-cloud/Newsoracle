"""
training/plot_training.py
────────────────────────────────────────────────────────────────────────────────
학습 결과를 시각화하여 PNG 이미지로 저장한다.

[생성 그래프 4종]
  1. train_loss vs val_loss 곡선 (과적합 감지용)
  2. F1-score 곡선 (epoch별 성능 추이)
  3. precision / recall 곡선 (클래스 균형 확인)
  4. 종합 대시보드 (4개 그래프 합본)

[호출 방법]
  trainer.py에서 학습 완료 후:
    from training.plot_training import save_training_plots
    save_training_plots(history, save_dir="models/saved/")

  또는 독립 실행:
    python -m training.plot_training --history models/saved/training_history.json

[history 형식]
  {
    "epochs": [1, 2, 3, ...],
    "train_loss": [0.68, 0.12, ...],
    "val_loss": [0.65, 0.18, ...],
    "f1": [0.85, 0.94, ...],
    "accuracy": [0.87, 0.95, ...],
    "precision": [0.82, 0.93, ...],
    "recall": [0.88, 0.96, ...],
    "best_epoch": 4,
    "best_f1": 0.9869,
    "early_stopped_at": 7
  }

[왜 이미지로 저장하는가]
  - 서버/터미널에서 직접 그래프를 볼 수 없는 환경 (GPU 서버 SSH 등)
  - README, 발표 자료, 프로젝트 문서에 포함 가능
  - scp로 로컬에 가져와서 확인 가능
"""

import os
import json
from logger import get_logger

logger = get_logger("plot_training")


def save_training_plots(history: dict, save_dir: str = "models/saved/") -> list:
    """
    학습 이력 데이터를 기반으로 그래프 4종을 PNG 이미지로 저장한다.

    Args:
        history: 학습 이력 딕셔너리
            필수 키: epochs, train_loss, val_loss, f1
            선택 키: accuracy, precision, recall, best_epoch, early_stopped_at
        save_dir: 이미지 저장 디렉토리

    Returns:
        저장된 이미지 파일 경로 리스트
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # GUI 없는 환경(서버)에서도 동작하도록
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        logger.warning(
            "matplotlib 미설치. 학습 그래프를 생성하려면: "
            "pip install matplotlib --break-system-packages"
        )
        return []

    os.makedirs(save_dir, exist_ok=True)
    saved_files = []

    # ── 한글 폰트 설정 (없으면 영어로 폴백) ─────────────────
    # Ubuntu에서 나눔고딕 폰트가 있으면 사용, 없으면 기본 폰트
    korean_font = None
    for font_name in ["NanumGothic", "NanumBarunGothic", "Malgun Gothic", "AppleGothic"]:
        try:
            font_path = fm.findfont(fm.FontProperties(family=font_name))
            if font_path and "LastResort" not in font_path:
                korean_font = font_name
                break
        except Exception:
            continue

    if korean_font:
        plt.rcParams["font.family"] = korean_font
        plt.rcParams["axes.unicode_minus"] = False
        logger.info(f"한글 폰트 설정: {korean_font}")
    else:
        logger.info("한글 폰트 없음 → 영어로 그래프 생성")

    # ── 데이터 추출 ──────────────────────────────────────────
    epochs = history.get("epochs", [])
    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss", [])
    f1_scores = history.get("f1", [])
    accuracy = history.get("accuracy", [])
    precision = history.get("precision", [])
    recall = history.get("recall", [])
    best_epoch = history.get("best_epoch", None)
    early_stopped = history.get("early_stopped_at", None)

    if not epochs:
        logger.warning("학습 이력이 비어있어 그래프를 생성할 수 없음")
        return []

    # ── 색상 정의 (하늘색 테마) ───────────────────────────────
    color_train = "#0ea5e9"    # sky-500
    color_val = "#f97316"      # orange-500
    color_f1 = "#22c55e"       # green-500
    color_acc = "#8b5cf6"      # violet-500
    color_prec = "#0284c7"     # sky-600
    color_recall = "#16a34a"   # green-600
    color_best = "#ef4444"     # red-500

    use_korean = korean_font is not None

    # ────────────────────────────────────────────────────────
    # 그래프 1: Loss 곡선 (과적합 감지)
    # ────────────────────────────────────────────────────────
    try:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, train_loss, marker="o", color=color_train, linewidth=2,
                label="Train Loss", markersize=6)
        ax.plot(epochs, val_loss, marker="s", color=color_val, linewidth=2,
                label="Val Loss", markersize=6)

        if best_epoch and best_epoch in epochs:
            idx = epochs.index(best_epoch)
            ax.axvline(x=best_epoch, color=color_best, linestyle="--", alpha=0.5,
                       label=f"Best Epoch ({best_epoch})")

        if early_stopped:
            ax.axvline(x=early_stopped, color="#94a3b8", linestyle=":", alpha=0.5,
                       label=f"Early Stop ({early_stopped})")

        ax.set_xlabel("Epoch" if not use_korean else "에포크", fontsize=12)
        ax.set_ylabel("Loss", fontsize=12)
        ax.set_title("Training Loss vs Validation Loss" if not use_korean
                      else "학습 손실 vs 검증 손실", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)
        fig.tight_layout()

        path = os.path.join(save_dir, "loss_curve.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)
        logger.info(f"Loss 곡선 저장: {path}")
    except Exception as e:
        logger.error(f"Loss 곡선 생성 실패: {e}")

    # ────────────────────────────────────────────────────────
    # 그래프 2: F1-score 곡선
    # ────────────────────────────────────────────────────────
    try:
        if f1_scores:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(epochs[:len(f1_scores)], f1_scores, marker="D", color=color_f1,
                    linewidth=2, markersize=8, label="F1-Score")

            if best_epoch and best_epoch in epochs:
                idx = epochs.index(best_epoch)
                if idx < len(f1_scores):
                    ax.annotate(
                        f"Best: {f1_scores[idx]:.4f}",
                        xy=(best_epoch, f1_scores[idx]),
                        xytext=(best_epoch + 0.3, f1_scores[idx] - 0.02),
                        fontsize=10, color=color_best, fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=color_best),
                    )

            ax.set_xlabel("Epoch" if not use_korean else "에포크", fontsize=12)
            ax.set_ylabel("F1-Score", fontsize=12)
            ax.set_title("F1-Score by Epoch" if not use_korean
                          else "에포크별 F1-Score", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(epochs[:len(f1_scores)])
            fig.tight_layout()

            path = os.path.join(save_dir, "f1_curve.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)
            logger.info(f"F1 곡선 저장: {path}")
    except Exception as e:
        logger.error(f"F1 곡선 생성 실패: {e}")

    # ────────────────────────────────────────────────────────
    # 그래프 3: Precision / Recall 곡선
    # ────────────────────────────────────────────────────────
    try:
        if precision and recall:
            fig, ax = plt.subplots(figsize=(8, 5))
            ep = epochs[:len(precision)]
            ax.plot(ep, precision, marker="^", color=color_prec, linewidth=2,
                    markersize=7, label="Precision")
            ax.plot(ep[:len(recall)], recall, marker="v", color=color_recall,
                    linewidth=2, markersize=7, label="Recall")

            ax.set_xlabel("Epoch" if not use_korean else "에포크", fontsize=12)
            ax.set_ylabel("Score", fontsize=12)
            ax.set_title("Precision / Recall by Epoch" if not use_korean
                          else "에포크별 정밀도 / 재현율", fontsize=14, fontweight="bold")
            ax.set_ylim(0, 1.05)
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_xticks(ep)
            fig.tight_layout()

            path = os.path.join(save_dir, "precision_recall_curve.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            saved_files.append(path)
            logger.info(f"Precision/Recall 곡선 저장: {path}")
    except Exception as e:
        logger.error(f"Precision/Recall 곡선 생성 실패: {e}")

    # ────────────────────────────────────────────────────────
    # 그래프 4: 종합 대시보드 (2x2)
    # ────────────────────────────────────────────────────────
    try:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # [0,0] Loss
        ax = axes[0][0]
        ax.plot(epochs, train_loss, marker="o", color=color_train, linewidth=2, label="Train Loss")
        ax.plot(epochs, val_loss, marker="s", color=color_val, linewidth=2, label="Val Loss")
        if best_epoch and best_epoch in epochs:
            ax.axvline(x=best_epoch, color=color_best, linestyle="--", alpha=0.5)
        ax.set_title("Loss" if not use_korean else "손실", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(epochs)

        # [0,1] F1
        ax = axes[0][1]
        if f1_scores:
            ax.plot(epochs[:len(f1_scores)], f1_scores, marker="D", color=color_f1, linewidth=2, label="F1")
            ax.set_ylim(0, 1.05)
        ax.set_title("F1-Score", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # [1,0] Accuracy
        ax = axes[1][0]
        if accuracy:
            ax.plot(epochs[:len(accuracy)], accuracy, marker="o", color=color_acc, linewidth=2, label="Accuracy")
            ax.set_ylim(0, 1.05)
        ax.set_title("Accuracy" if not use_korean else "정확도", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # [1,1] Precision / Recall
        ax = axes[1][1]
        if precision:
            ax.plot(epochs[:len(precision)], precision, marker="^", color=color_prec, linewidth=2, label="Precision")
        if recall:
            ax.plot(epochs[:len(recall)], recall, marker="v", color=color_recall, linewidth=2, label="Recall")
        ax.set_ylim(0, 1.05)
        ax.set_title("Precision / Recall" if not use_korean
                      else "정밀도 / 재현율", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # 전체 타이틀
        best_f1 = history.get("best_f1", max(f1_scores) if f1_scores else 0)
        fig.suptitle(
            f"Newsoracle Training Dashboard | Best F1={best_f1:.4f} (epoch={best_epoch})"
            if not use_korean else
            f"Newsoracle 학습 대시보드 | 최고 F1={best_f1:.4f} (에포크={best_epoch})",
            fontsize=15, fontweight="bold", y=1.02,
        )
        fig.tight_layout()

        path = os.path.join(save_dir, "training_dashboard.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        saved_files.append(path)
        logger.info(f"종합 대시보드 저장: {path}")
    except Exception as e:
        logger.error(f"종합 대시보드 생성 실패: {e}")

    # ── history를 JSON으로도 저장 (나중에 재생성 가능) ─────────
    try:
        history_path = os.path.join(save_dir, "training_history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)
        saved_files.append(history_path)
        logger.info(f"학습 이력 JSON 저장: {history_path}")
    except Exception as e:
        logger.error(f"학습 이력 JSON 저장 실패: {e}")

    logger.info(f"학습 그래프 생성 완료 | {len(saved_files)}개 파일 저장")
    return saved_files


# ────────────────────────────────────────────────────────────
# CLI 독립 실행 (기존 history JSON으로 그래프 재생성)
# ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description="학습 그래프 생성")
    parser.add_argument(
        "--history",
        type=str,
        default="models/saved/training_history.json",
        help="학습 이력 JSON 파일 경로",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="models/saved/",
        help="그래프 이미지 저장 디렉토리",
    )
    args = parser.parse_args()

    if not os.path.exists(args.history):
        print(f"파일을 찾을 수 없음: {args.history}")
        sys.exit(1)

    with open(args.history, "r", encoding="utf-8") as f:
        history = json.load(f)

    files = save_training_plots(history, save_dir=args.output)

    if files:
        print(f"\n생성된 파일 {len(files)}개:")
        for fp in files:
            print(f"  {fp}")
    else:
        print("그래프 생성 실패. matplotlib가 설치되었는지 확인해주세요.")
