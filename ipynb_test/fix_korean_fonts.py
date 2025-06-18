#!/usr/bin/env python3
"""
Jupyter 노트북에 한글 폰트 설정을 추가하는 스크립트
"""

import json
import os

def add_korean_font_setup(notebook_path):
    """노트북에 한글 폰트 설정 셀을 추가"""
    
    # 노트북 파일 읽기
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)
    
    print("🎨 한글 폰트 설정 추가 중...")
    
    # 한글 폰트 설정 셀
    font_setup_cell = {
        "cell_type": "code",
        "execution_count": None,
        "id": "korean_font_setup",
        "metadata": {},
        "outputs": [],
        "source": [
            "# 한글 폰트 설정 - 폰트 깨짐 해결\n",
            "import matplotlib.pyplot as plt\n",
            "import matplotlib.font_manager as fm\n",
            "import platform\n",
            "\n",
            "def setup_korean_fonts():\n",
            "    \"\"\"운영체제별 한글 폰트 설정\"\"\"\n",
            "    system = platform.system()\n",
            "    \n",
            "    # 사용 가능한 한글 폰트 리스트\n",
            "    korean_fonts = [\n",
            "        'Malgun Gothic',     # Windows\n",
            "        'AppleGothic',       # macOS\n",
            "        'Apple SD Gothic Neo', # macOS\n",
            "        'NanumGothic',       # Linux/Windows\n",
            "        'NanumBarunGothic',  # 일반적\n",
            "        'DejaVu Sans',       # 대체 폰트\n",
            "    ]\n",
            "    \n",
            "    # 시스템에서 사용 가능한 폰트 찾기\n",
            "    available_fonts = [f.name for f in fm.fontManager.ttflist]\n",
            "    \n",
            "    selected_font = None\n",
            "    for font in korean_fonts:\n",
            "        if font in available_fonts:\n",
            "            selected_font = font\n",
            "            break\n",
            "    \n",
            "    if selected_font:\n",
            "        plt.rcParams['font.family'] = selected_font\n",
            "        print(f\"✅ 한글 폰트 설정 완료: {selected_font}\")\n",
            "    else:\n",
            "        # 폰트를 찾지 못한 경우 기본 설정\n",
            "        if system == 'Darwin':  # macOS\n",
            "            plt.rcParams['font.family'] = 'AppleGothic'\n",
            "        elif system == 'Windows':\n",
            "            plt.rcParams['font.family'] = 'Malgun Gothic'\n",
            "        else:  # Linux\n",
            "            plt.rcParams['font.family'] = 'DejaVu Sans'\n",
            "        print(f\"⚠️  기본 폰트 사용: {plt.rcParams['font.family']}\")\n",
            "    \n",
            "    # 마이너스 기호 깨짐 방지\n",
            "    plt.rcParams['axes.unicode_minus'] = False\n",
            "    \n",
            "    # 폰트 캐시 새로고침\n",
            "    plt.rcParams['font.size'] = 10\n",
            "    \n",
            "    return selected_font\n",
            "\n",
            "# 한글 폰트 설정 실행\n",
            "font_name = setup_korean_fonts()\n",
            "\n",
            "# 테스트 플롯\n",
            "import numpy as np\n",
            "fig, ax = plt.subplots(figsize=(8, 4))\n",
            "x = np.arange(4)\n",
            "values = [0.8, 0.6, 0.7, 0.9]\n",
            "labels = ['TF-IDF', 'LSA', 'Word2Vec', '정규화된\\n가중합']\n",
            "\n",
            "bars = ax.bar(x, values, color=['skyblue', 'lightgreen', 'orange', 'pink'], alpha=0.8)\n",
            "ax.set_xlabel('모델 타입')\n",
            "ax.set_ylabel('성능 점수')\n",
            "ax.set_title('🎯 한글 폰트 테스트 - 모델 성능 비교')\n",
            "ax.set_xticks(x)\n",
            "ax.set_xticklabels(labels)\n",
            "\n",
            "# 막대 위에 값 표시\n",
            "for bar, value in zip(bars, values):\n",
            "    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,\n",
            "            f'{value:.2f}', ha='center', va='bottom')\n",
            "\n",
            "plt.grid(True, alpha=0.3)\n",
            "plt.tight_layout()\n",
            "plt.show()\n",
            "\n",
            "print(\"\\n🎨 한글 폰트 설정이 완료되었습니다!\")\n",
            "print(f\"📊 이제 모든 차트에서 한글이 정상적으로 표시됩니다.\")"
        ]
    }
    
    # 기존 한글 폰트 설정이 있는지 확인
    has_font_setup = False
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
            if 'setup_korean_fonts' in source or 'korean_font_setup' in cell.get('id', ''):
                has_font_setup = True
                break
    
    if not has_font_setup:
        # 라이브러리 import 셀 바로 다음에 추가
        insert_index = -1
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code' and 'source' in cell:
                source = ''.join(cell['source']) if isinstance(cell['source'], list) else cell['source']
                if 'import matplotlib.pyplot as plt' in source:
                    insert_index = i + 1
                    break
        
        if insert_index > -1:
            notebook['cells'].insert(insert_index, font_setup_cell)
        else:
            # import 셀을 찾지 못한 경우 처음에 추가
            notebook['cells'].insert(0, font_setup_cell)
    
    # 수정된 노트북 저장
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    
    if not has_font_setup:
        print(f"✅ 한글 폰트 설정 셀이 추가되었습니다: {notebook_path}")
        print("🎨 추가된 기능:")
        print("   • 운영체제별 자동 한글 폰트 감지")
        print("   • 마이너스 기호 깨짐 방지")
        print("   • 폰트 설정 테스트 차트")
    else:
        print("ℹ️ 한글 폰트 설정이 이미 존재합니다.")
    
    return True

def main():
    """메인 함수"""
    notebook_path = 'balanced_training_demo.ipynb'
    
    if not os.path.exists(notebook_path):
        print(f"❌ 노트북 파일을 찾을 수 없습니다: {notebook_path}")
        return False
    
    print(f"🚀 한글 폰트 설정 추가 시작: {notebook_path}")
    
    try:
        success = add_korean_font_setup(notebook_path)
        if success:
            print("\n🎉 완료! 이제 노트북에서 한글이 정상적으로 표시됩니다.")
            print("💡 실행 방법:")
            print("   1. Jupyter 노트북 열기")
            print("   2. '한글 폰트 설정' 셀 실행")
            print("   3. 테스트 차트로 한글 표시 확인")
            print("   4. 이후 모든 차트에서 한글 정상 표시")
        return success
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main() 