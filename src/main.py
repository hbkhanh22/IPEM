import argparse
from animal_classifier import AnimalImageClassifier
from caltech_classifier import CaltechImageClassifier
from brain_tumor_classifier import BrainTumorClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, choices=['animals', 'caltech-101', 'brain-tumor'], default='animals')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--model', type=str, choices=["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3", "vgg19", "transformer"], default="efficientnet_b3")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    parser.add_argument('--mode', type=str, choices=["train", "test", "explain"], default="test")
    parser.add_argument('--test-loader-fraction', type=float, default=1.0)
    return parser.parse_args()

def main():
    args = parse_args()
    if args.dataset.lower() == 'animals' or args.dataset.lower() == 'animal':
        clf = AnimalImageClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs,
                          test_loader_fraction=args.test_loader_fraction)
    elif args.dataset.lower() == 'caltech-101':
        clf = CaltechImageClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs,
                          test_loader_fraction=args.test_loader_fraction)
    elif args.dataset.lower() == 'brain-tumor':
        clf = BrainTumorClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs,
                          test_loader_fraction=args.test_loader_fraction)
    
    if args.mode == "train":
        clf.train()
    elif args.mode == "test":
        clf.test()
    elif args.mode == "explain":
        clf.load_trained_model()
        clf._build_dataloaders()

        print(f"Choosing your explanation method: \n1. LIME\n2. SHAP\n3. IPEM\n4. GradCAM")
        method = input("Enter the number of the explanation method: ")
        if method == "1":
            lime_results = clf.run_lime_metrics()
        elif method == "2":
            shap_results = clf.run_shap_metrics()
        elif method == "3":
            ipem_results = clf.run_ipem_metrics()
        elif method == "4":
            gradcam_results = clf.run_gradcam_metrics()
        elif method == "5":
            rise_results = clf.run_rise_metrics()

if __name__ == "__main__":
    main()