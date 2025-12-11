from ultralytics import YOLO
import torch
import torch.nn.utils.prune as prune
import os

# --- CẤU HÌNH ---
MODEL_PATH = '/Users/haminhanh/yolov11_nano1216.pt'
'''AMOUNT_TO_PRUNE = 0.3  # Cắt tỉa 30% các kết nối có trọng số nhỏ nhất
Thử 50% '''
AMOUNT_TO_PRUNE = 0.5
# ----------------

def main():
    print(f"🔄 Đang tải mô hình: {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    
    # Truy cập vào mô hình PyTorch gốc bên trong wrapper của Ultralytics
    pytorch_model = model.model
    
    print(f"✂️ Bắt đầu cắt tỉa (Unstructured L1 Pruning) với tỷ lệ {AMOUNT_TO_PRUNE*100}%...")
    
    # 1. Duyệt qua tất cả các lớp và áp dụng Pruning cho Conv2d
    parameters_to_prune = []
    for name, module in pytorch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            # Chọn cắt tỉa trọng số (weight) của lớp Conv2d
            parameters_to_prune.append((module, 'weight'))
    
    # Áp dụng Global Pruning (Cắt tỉa toàn cục)
    # Loại bỏ 30% trọng số nhỏ nhất trên TOÀN BỘ mạng (không phải từng lớp riêng lẻ)
    # Điều này tốt hơn vì nó giữ lại các lớp quan trọng.
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=AMOUNT_TO_PRUNE,
    )
    
    # 2. "Cam kết" việc cắt tỉa (Làm cho nó vĩnh viễn)
    # Bước này loại bỏ các mask tạm thời và ghi đè trọng số bằng 0
    for module, _ in parameters_to_prune:
        prune.remove(module, 'weight')
        
    print("✅ Cắt tỉa hoàn tất.")
    
    # 3. Kiểm tra độ thưa (Sparsity)
    total_zeros = 0
    total_params = 0
    for name, module in pytorch_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            total_zeros += torch.sum(module.weight == 0)
            total_params += module.weight.nelement()
            
    print(f"📊 Thống kê sau khi cắt tỉa:")
    print(f"   - Tổng số tham số (Conv2d): {total_params}")
    print(f"   - Số tham số bằng 0: {total_zeros}")
    print(f"   - Độ thưa (Sparsity): {100. * total_zeros / total_params:.2f}%")
    
    # 4. Lưu mô hình đã cắt tỉa
    save_path = MODEL_PATH.replace('.pt', '_pruned.pt')
    
    # Lưu chỉ dict trọng số (state_dict) hoặc cả mô hình để Ultralytics có thể load lại
    # Lưu ý: Ultralytics có cơ chế lưu riêng, nhưng ta sẽ dùng torch.save để đảm bảo cấu trúc
    torch.save(model.ckpt, save_path)
    print(f"💾 Đã lưu mô hình cắt tỉa tại: {save_path}")
    print("\n⚠️ LƯU Ý: Bạn cần Fine-tune (train lại) mô hình này để phục hồi độ chính xác!")

if __name__ == "__main__":
    main()