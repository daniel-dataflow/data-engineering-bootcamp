import torch

# MPS 사용 가능 여부 확인
if torch.backends.mps.is_available():
    print("축하합니다! M1 GPU(MPS)를 사용할 수 있습니다.")
    
    # MPS 장치를 변수로 설정
    device = torch.device("mps")
    
    # 간단한 텐서 연산을 MPS에서 수행 테스트
    x = torch.randn(3, 3, device=device)
    print("MPS 장치에서 텐서가 성공적으로 생성되었습니다:")
    print(x)
    
else:
    print("아쉽지만 MPS를 사용할 수 없습니다.")
    print("PyTorch(1.12 이상)와 macOS(12.3 이상) 버전을 확인해주세요.")