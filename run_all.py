"""
run_all.py — Master script to run the full project pipeline.
Run individual tasks with:
  python run_all.py --task a
  python run_all.py --task b
  python run_all.py --task c
  python run_all.py --task d
  python run_all.py --task all
"""
import argparse
import sys
import os


def run_task_a():
    print("\n" + "="*60)
    print("TASK A — CNN Baseline and Transfer Learning")
    print("="*60)

    from task_a.train import train

    # 1. Train custom CNN with each augmentation level
    for aug in ["light", "medium", "heavy"]:
        train(model_name="custom", aug_intensity=aug)

    # 2. Train transfer learning models
    for model in ["resnet50", "efficientnet_b3", "mobilenet_v2"]:
        train(model_name=model, aug_intensity="medium")

    # 3. Ablation study (quick, 5 epochs each)
    from task_a.ablation import run_ablation
    run_ablation(ablation_epochs=5)

    # 4. Grad-CAM for best models
    from task_a.gradcam import visualize_gradcam_batch
    import yaml
    from torchvision import datasets
    with open("configs/config.yaml") as f:
        import yaml; cfg = yaml.safe_load(f)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    num_classes = cfg["data"]["num_classes"]
    ds = datasets.ImageFolder(cfg["data"]["val_dir"])

    for mname in ["custom", "resnet50"]:
        from task_a.custom_cnn import CustomCNN
        from task_a.transfer_learning import get_model
        import torch, os

        ckpt = os.path.join(cfg["paths"]["save_dir"], f"{mname}_best.pt")
        if not os.path.exists(ckpt):
            continue

        if mname == "custom":
            model = CustomCNN(num_classes)
        else:
            model = get_model(mname, num_classes)

        model.load_state_dict(torch.load(ckpt, map_location=device))
        model = model.to(device)

        visualize_gradcam_batch(
            model=model, model_name=mname,
            val_dir=cfg["data"]["val_dir"],
            class_names=ds.classes, device=device,
            n_images=8,
            save_path=os.path.join(cfg["paths"]["log_dir"], f"{mname}_gradcam.png"),
            image_size=cfg["data"]["image_size"],
        )

    # 5. Edge deployment / ONNX export
    from task_a.export_onnx import run_edge_analysis
    run_edge_analysis("resnet50")


def run_task_b():
    print("\n" + "="*60)
    print("TASK B — GANs and Diffusion Models")
    print("="*60)

    # 1. DCGAN on minority class
    from task_b.dcgan import train_dcgan
    train_dcgan(target_class_idx=0)

    # 2. cGAN (all classes)
    from task_b.cgan import train_cgan
    train_cgan()

    # 3. DDPM
    from task_b.ddpm import train_ddpm
    train_ddpm()

    # 4. FID + IS scores
    from task_b.fid_score import compute_all_metrics
    compute_all_metrics()

    # 5. Downstream impact evaluation
    from task_b.downstream import run_downstream_evaluation
    run_downstream_evaluation()


def run_task_c():
    print("\n" + "="*60)
    print("TASK C — Vision Transformers")
    print("="*60)

    from task_c.vit_timm import train_vit

    # 1. ViT-B/16 with pretrained weights
    train_vit("vit_b16",    pretrained=True,  use_mixup=False)

    # 2. ViT-B/16 WITHOUT pretrained (for comparison)
    train_vit("vit_b16",    pretrained=False, use_mixup=False)

    # 3. DeiT-Small
    train_vit("deit_small", pretrained=True,  use_mixup=True)

    # 4. Swin-Tiny
    train_vit("swin_tiny",  pretrained=True,  use_mixup=False)

    # 5. Hybrid CNN-ViT
    train_vit("hybrid",     pretrained=True,  use_mixup=False)

    # 6. Attention visualisation
    from task_c.attention_viz import visualize_attention
    for model_name in ["vit_b16", "deit_small", "swin_tiny"]:
        visualize_attention(model_name, n_images=8)

    # 7. Full benchmark table
    from task_c.benchmark import build_benchmark_table
    build_benchmark_table()


def run_task_d():
    print("\n" + "="*60)
    print("TASK D — Integration, Deployment & Analysis")
    print("="*60)

    # 1. Optimise ensemble weights
    from task_d.ensemble import optimise_ensemble_weights
    optimise_ensemble_weights()

    # 2. Calibration (ECE + reliability diagrams)
    from task_d.calibration import run_calibration_analysis
    run_calibration_analysis()

    # 3. Distribution shift + energy analysis
    from task_d.deployment_analysis import run_deployment_analysis
    run_deployment_analysis()

    print("\n  To start the API server:")
    print("    python -m task_d.api")
    print("  Or with Docker:")
    print("    docker-compose up --build")


def main():
    parser = argparse.ArgumentParser(
        description="CS4152 Smart Agriculture Project Runner"
    )
    parser.add_argument(
        "--task",
        default="all",
        choices=["a", "b", "c", "d", "all"],
        help="Which task to run"
    )
    args = parser.parse_args()

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs",        exist_ok=True)
    os.makedirs("samples",     exist_ok=True)

    if args.task in ("a", "all"):
        run_task_a()
    if args.task in ("b", "all"):
        run_task_b()
    if args.task in ("c", "all"):
        run_task_c()
    if args.task in ("d", "all"):
        run_task_d()

    print("\n" + "="*60)
    print("All done. Check logs/ for plots and checkpoints/ for models.")
    print("="*60)


if __name__ == "__main__":
    main()
