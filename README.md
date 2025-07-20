# 🎮 PyJoySim

**조이스틱을 이용한 파이썬 시뮬레이션 프로그램 모음**

PyJoySim은 교육용 및 연구용 시뮬레이션을 위한 통합 플랫폼입니다. 조이스틱으로 직관적으로 조작할 수 있는 다양한 시뮬레이션을 제공하여, 물리학, 로보틱스, 항공역학 등을 재미있게 학습할 수 있습니다.

## ✨ 주요 기능

### 🚗 차량 시뮬레이션

- 현실적인 2D 자동차 물리 모델
- 다양한 트랙과 차량 종류
- 충돌 감지 및 손상 시스템

### 🤖 로봇 시뮬레이션

- 3-DOF 로봇 팔 제어
- 역기구학 계산 및 시각화
- 정밀 제어 및 작업 공간 표시

### 🚁 드론 시뮬레이션 (Phase 3)

- 3D 쿼드로터 비행 물리
- 다양한 비행 모드 (수동, 안정화, 자동)
- 바람 효과 및 센서 시뮬레이션

### 🚀 우주선 시뮬레이션 (Phase 3)

- 무중력 환경 물리
- 궤도 역학 및 연료 관리
- 행성 간 항행 및 도킹

### 🚢 잠수함 시뮬레이션 (Phase 3)

- 수중 물리 환경
- 부력 및 압력 계산
- 소나 시스템 및 해저 탐사

## 🎯 대상 사용자

- **교육 기관**: 물리학, 공학 교육용 도구
- **연구자**: 시뮬레이션 프로토타이핑
- **학생**: 재미있는 STEM 학습
- **개발자**: 게임 및 시뮬레이션 개발 학습

## 🛠️ 설치 방법

### 요구사항

- **Python**: 3.8 이상
- **조이스틱**: Xbox Controller, PS4/PS5 Controller 등
- **운영체제**: Windows 10/11, macOS 10.14+, Ubuntu 18.04+

### 개발 환경 설치

1. **저장소 클론**

   ```bash
   git clone https://github.com/ai-research/pyjoysim.git
   cd pyjoysim
   ```

2. **uv를 사용한 의존성 설치** (권장)

   ```bash
   # uv 설치 (아직 없다면)
   pip install uv
   
   # 프로젝트 의존성 설치
   uv sync
   ```

3. **또는 pip 사용**

   ```bash
   pip install -e .
   ```

4. **조이스틱 연결 및 실행**

   ```bash
   python main.py
   ```

## 🚀 빠른 시작

1. **조이스틱 연결**: USB 또는 Bluetooth로 컨트롤러 연결
2. **시뮬레이션 선택**: 메인 메뉴에서 원하는 시뮬레이션 선택
3. **튜토리얼**: 각 시뮬레이션의 내장 튜토리얼 따라하기
4. **자유 플레이**: 다양한 설정으로 실험해보기

## 📁 프로젝트 구조

```text
pyjoysim/
├── pyjoysim/                 # 메인 패키지
│   ├── core/                 # 핵심 시스템
│   ├── input/                # 조이스틱 입력 관리
│   ├── simulation/           # 시뮬레이션 모듈들
│   │   ├── vehicle/          # 차량 시뮬레이션
│   │   ├── robot/            # 로봇 시뮬레이션
│   │   └── game/             # 게임형 시뮬레이션
│   ├── physics/              # 물리 엔진
│   ├── rendering/            # 렌더링 시스템
│   ├── ui/                   # 사용자 인터페이스
│   └── config/               # 설정 관리
├── tests/                    # 테스트 코드
├── examples/                 # 예제 프로그램
├── assets/                   # 게임 에셋
└── docs/                     # 문서
```

## 📖 개발 진행 상황

PyJoySim은 4단계에 걸쳐 개발되고 있습니다:

| 단계 | 상태 | 기능 |
|------|------|------|
| **Phase 1** | 🚧 진행중 | 기반 시스템 (입력, 물리, 렌더링) |
| **Phase 2** | 📋 계획됨 | 기본 시뮬레이션 (자동차, 로봇팔) |
| **Phase 3** | 📋 계획됨 | 고급 기능 (3D, 드론, 우주선) |
| **Phase 4** | 📋 계획됨 | 최적화 및 배포 |

자세한 개발 계획은 [`docs/development-roadmap.md`](docs/development-roadmap.md)를 참조하세요.

## 🎮 지원 조이스틱

### ✅ 완전 지원

- Xbox Series X|S Controller
- Xbox One Controller  
- PlayStation 5 DualSense
- PlayStation 4 DualShock 4

### 🔄 부분 지원

- Nintendo Switch Pro Controller
- Generic USB Gamepad

### 📝 추가 예정

- Steam Controller
- 8BitDo Controllers
- 커스텀 조이스틱

## 🧪 개발 및 기여

### 개발 환경 설정

```bash
# 개발 의존성 포함 설치
uv sync --dev

# 테스트 실행
pytest

# 코드 품질 검사
ruff check .
ruff format .
```

### 기여 방법

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

자세한 기여 가이드라인은 [`CONTRIBUTING.md`](CONTRIBUTING.md)를 참조하세요.

## 📋 할 일 목록

### Phase 1 (현재)
- [x] 프로젝트 구조 설정
- [x] 개발 계획 문서 작성
- [ ] pyproject.toml 설정 완성
- [ ] 조이스틱 입력 시스템 구현
- [ ] 기본 물리 엔진 통합
- [ ] 2D 렌더링 시스템

### 향후 계획
- [ ] 자동차 시뮬레이션 완성
- [ ] 로봇 팔 시뮬레이션 구현
- [ ] 3D 렌더링 시스템
- [ ] 드론 비행 시뮬레이션
- [ ] 멀티플레이어 지원

## 📚 문서

- [📋 기술 요구사항 명세 (TRD)](docs/TRD.md)
- [🗺️ 개발 로드맵](docs/development-roadmap.md)
- [🏗️ Phase 1: 기반 시스템](docs/phase1-foundation.md)
- [🚗 Phase 2: 기본 시뮬레이션](docs/phase2-basic-simulations.md)
- [🚁 Phase 3: 고급 기능](docs/phase3-advanced-features.md)
- [🎯 Phase 4: 완성 및 배포](docs/phase4-completion.md)

## 🔧 기술 스택

### 현재 (Phase 1-2)

- **언어**: Python 3.8+
- **물리**: pymunk (2D)
- **그래픽**: pygame
- **의존성**: uv
- **테스트**: pytest

### 계획 (Phase 3-4)

- **3D 물리**: PyBullet
- **3D 그래픽**: ModernGL
- **네트워킹**: asyncio, websockets
- **패키징**: PyInstaller

## 📈 성능 목표

- **프레임레이트**: 최소 30 FPS, 목표 60 FPS
- **입력 지연**: 50ms 이하
- **메모리 사용량**: 1GB 이하 (기본 시뮬레이션)
- **시작 시간**: 5초 이하

## 📄 라이센스

이 프로젝트는 MIT 라이센스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

## 🤝 지원 및 문의

- **GitHub Issues**: 버그 리포트 및 기능 요청
- **GitHub Discussions**: 일반적인 질문 및 토론
- **Email**: [이메일 주소]

## 🙏 감사의 말

이 프로젝트는 교육용 시뮬레이션의 발전을 위해 시작되었습니다. 기여해 주시는 모든 분들께 감사드립니다.

---

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요!**
