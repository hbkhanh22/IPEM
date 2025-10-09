import numpy as np
import torch
import argparse

from covid_classifier import COVIDClassifier
from animal_classifier import AnimalImageClassifier
from caltech_classifier import CaltechImageClassifier
from brain_tumor_classifier import BrainTumorClassifier
from sample_prediction import load_sample_from_animals, load_sample_from_brain_tumor, load_sample_from_caltech, load_sample_from_covid, predict_with_model, explain_with_lime, explain_with_pebex, explain_with_shap


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main(args_dataset, args_model_name):
    # Cấu hình
    data_dir = "data"  # có thể là "data" hoặc "data/animals"
    output_dir = "outputs"  # Sử dụng thư mục outputs chính
    sample_output_dir = "outputs/sample_explanation"  # Thư mục cho kết quả mẫu
    img_size = 224
    
    print("🚀 Bắt đầu script...")
    
    try:
        if args_dataset.lower() == 'animals': 
            clf = AnimalImageClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)

            # Lấy mẫu ngẫu nhiên từ tập động vật
            print("\n🖼️  Lấy mẫu ngẫu nhiên từ dataset động vật...")
            img_tensor, img_np, filename, org_img, selected_class = load_sample_from_animals(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")
        elif args_dataset.lower() == 'sars-cov2-ct':
        # Khởi tạo classifier
            clf = COVIDClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)

            # Lấy mẫu từ COVID
            print("\n🖼️  Lấy mẫu từ tập COVID...")
            img_tensor, img_np, filename, org_img = load_sample_from_covid(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")
        elif args_dataset.lower() == 'caltech-101':
            clf = CaltechImageClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)
            # Lấy mẫu từ Caltech-101
            print("\n🖼️  Lấy mẫu từ tập Caltech-101...")
            img_tensor, img_np, filename, org_img, selected_class = load_sample_from_caltech(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")
        elif args_dataset.lower() == 'brain-tumor':
            clf = BrainTumorClassifier(data_dir=data_dir, output_dir=output_dir, img_size=img_size, args_model=args_model_name)
            # Lấy mẫu từ thư mục Testing trong brain-tumor
            print("\n🖼️  Lấy mẫu từ thư mục Testing của tập Brain-tumor...")
            img_tensor, img_np, filename, org_img, selected_class = load_sample_from_brain_tumor(data_dir, img_size)
            print(f"   - Ảnh gốc shape: {np.array(org_img).shape}")

        clf._build_dataloaders()
        clf.load_trained_model()
        
        print("📊 Thông tin mô hình:")
        print(f"   - Classes: {clf.class_names}")
        print(f"   - Device: {device}")
        
        # Set random seed
        np.random.seed(42)
        torch.manual_seed(42)
        
        # Dự đoán với mô hình
        print("\n🔮 Dự đoán với mô hình...")
        pred_class, confidence, probs = predict_with_model(clf.model, img_tensor)
        # Xác định index lớp đúng dựa trên tên class được chọn
        if selected_class in clf.class_names:
            true_class = clf.class_names.index(selected_class)
        else:
            raise ValueError(f"Lớp '{selected_class}' không nằm trong class_names của mô hình: {clf.class_names}")
        
        print(f"   - Ảnh: {filename}")
        print(f"   - True class: {clf.class_names[true_class]} (index: {true_class})")
        print(f"   - Predicted class: {clf.class_names[pred_class]} (index: {pred_class})")
        print(f"   - Confidence: {confidence:.4f}")
        print(f"   - Prediction correct: {pred_class == true_class}")
        
        # In xác suất cho tất cả classes
        print("\n📈 Xác suất dự đoán:")
        for i, (class_name, prob) in enumerate(zip(clf.class_names, probs)):
            print(f"   - {class_name}: {prob:.4f}")
        
        # Nếu dự đoán đúng, giải thích quyết định
        if pred_class == true_class:
            # LIME
            lime_explanation, lime_mask = explain_with_lime(clf, img_np, clf.class_names, sample_output_dir, args_dataset, org_img)
            # SHAP
            shap_values, shap_heatmap = explain_with_shap(clf, img_tensor, clf.class_names, sample_output_dir, args_dataset, org_img)
            # PEBEX
            pebex_heatmap = explain_with_pebex(clf, img_tensor, clf.class_names, sample_output_dir, args_dataset, org_img)
            
            print(f"\n💾 Kết quả đã được lưu trong: {sample_output_dir}")
            print("   - LIME: outputs/sample_explanation/lime/lime_explanation.png")
            print("   - SHAP: outputs/sample_explanation/shap/shap_explanation.png")
            print("   - PEBEX: outputs/sample_explanation/pebex/pebex_explanation.png")
            
        else:
            print(f"\n❌ Dự đoán sai! Không thể giải thích quyết định.")
            print("   Hãy thử với mẫu khác...")
        
    except Exception as e:
        print(f"❌ Lỗi: {e}")
        import traceback
        traceback.print_exc()
        input("Nhấn Enter để tiếp tục...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['animals', 'sars-cov2', 'caltech-101', 'brain-tumor'], default='animals')
    parser.add_argument('--model', type=str, choices=["efficientnet_b3", "resnet50", "densenet121", "mobilenet_v3", "vgg19"], default="efficientnet_b3")

    dataset = parser.parse_args().dataset
    model_name = parser.parse_args().model
    main(dataset, model_name)