import torch; model = torch.load('models/fixed_ocr_20250614_150024/best_fixed_model.pth', map_location='cpu', weights_only=False); print('🎯 TRAINING COMPLETE!'); print(f'Epochs: {model[\
epoch\]}'); print(f'Loss: {model[\validation_loss\]:.4f}'); print(f'Parameters: {len(model[\model_state_dict\])}'); print(f'Vocab: {model[\vocab_size\]} chars'); print('✅ Ready for production!')
