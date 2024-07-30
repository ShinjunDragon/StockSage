import pkgutil
import sys

def check_module(module_name):
    # 모듈이 설치되어 있는지 확인
    if pkgutil.find_loader(module_name) is not None:
        print(f"Module '{module_name}' is available.")
    else:
        print(f"Module '{module_name}' is not available.")

if __name__ == "__main__":
    # 확인할 모듈 이름을 여기에 입력하세요
    module_to_check = "StockSage"  # 예: "numpy", "pandas" 등
    check_module(module_to_check)
