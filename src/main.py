import argparse
from animal_classifier import AnimalImageClassifier
from covid_classifier import COVIDClassifier
from caltech_classifier import CaltechImageClassifier
from brain_tumor_classifier import BrainTumorClassifier

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--dataset', type=str, choices=['animals', 'sars-cov2', 'caltech-101', 'brain-tumor'], default='animals')
    parser.add_argument('--output-dir', type=str, default='outputs')
    parser.add_argument('--model', type=str, choices=["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3", "vgg19"], default="efficientnet_b3")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--img-size', type=int, default=224)
    return parser.parse_args()
    
def main():
    args = parse_args() 
    if args.dataset.lower() == 'animals' or args.dataset.lower() == 'animal':
        clf = AnimalImageClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs)
    elif args.dataset.lower() == 'sars-cov2':
        clf = COVIDClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs)
    elif args.dataset.lower() == 'caltech-101':
        clf = CaltechImageClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs)
    elif args.dataset.lower() == 'brain-tumor':
        clf = BrainTumorClassifier(data_dir=args.data_dir, output_dir=args.output_dir, args_model=args.model,
                          img_size=args.img_size, batch_size=args.batch_size, epochs=args.epochs)
    # clf.train()
    clf.test()
    #lime_results = clf.run_lime_metrics()
    #shap_results = clf.run_shap_metrics()
    pebex_results = clf.run_pebex_metrics()

if __name__ == "__main__":
    main()  