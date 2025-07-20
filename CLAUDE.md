# Claude 개발 컨텍스트

PyJoySim 프로젝트의 Claude Code 작업을 위한 컨텍스트 문서입니다.

## 프로젝트 개요

**PyJoySim**은 조이스틱을 이용한 교육용 파이썬 시뮬레이션 프로그램 모음입니다. 22주에 걸쳐 4단계로 개발되며, 현재 Phase 1 (기반 시스템 구축) 단계에 있습니다.

## 현재 개발 상태

### ✅ 완료된 작업

- [x] 프로젝트 디렉토리 구조 생성
- [x] TRD (기술 요구사항 명세) 작성
- [x] 4단계 개발 계획 문서 작성
- [x] README.md 업데이트
- [x] CLAUDE.md 생성

### 🚧 현재 진행 중 (Phase 1, Week 1)

- [ ] pyproject.toml 설정 완성
- [ ] 의존성 관리 및 가상환경 설정
- [ ] 기본 설정 시스템 구현
- [ ] 로깅 시스템 구현
- [ ] 예외 처리 기본 프레임워크

## 개발 단계별 계획

### Phase 1: 기반 시스템 (4주) - 현재 단계

**목표**: 조이스틱 입력, 물리 엔진, 렌더링 시스템 구축

#### Week 1: 프로젝트 설정

- pyproject.toml 설정
- 의존성 관리 (uv 사용)
- 기본 설정 및 로깅 시스템

#### Week 2: 조이스틱 입력 시스템

- JoystickManager 클래스
- InputProcessor 클래스  
- ConfigManager 클래스

#### Week 3: 물리 엔진 통합

- PhysicsEngine 추상 클래스
- Physics2D (pymunk 기반)
- 기본 물리 객체들

#### Week 4: 렌더링 시스템

- RenderEngine 추상 클래스
- Renderer2D (pygame 기반)
- BaseSimulation 프레임워크

### Phase 2: 기본 시뮬레이션 (6주)

- 2D 자동차 시뮬레이션
- 로봇 팔 시뮬레이션
- 기본 UI 구현

### Phase 3: 고급 기능 (8주)

- 3D 렌더링 시스템
- 드론 시뮬레이션
- 우주선/잠수함 시뮬레이션

### Phase 4: 완성 및 배포 (4주)

- 성능 최적화
- 버그 수정
- 문서화 및 배포

## 프로젝트 구조

```text
pyjoysim/
├── pyjoysim/           # 메인 패키지
│   ├── __init__.py     # 패키지 초기화
│   ├── core/           # 핵심 시스템
│   │   ├── __init__.py
│   │   ├── simulation_manager.py    # [구현 필요]
│   │   └── base_simulation.py       # [구현 필요]
│   ├── input/          # 조이스틱 입력 관리
│   │   ├── __init__.py
│   │   ├── joystick_manager.py      # [구현 필요]
│   │   ├── input_processor.py       # [구현 필요]
│   │   └── config_manager.py        # [구현 필요]
│   ├── simulation/     # 시뮬레이션 모듈들
│   │   ├── __init__.py
│   │   ├── vehicle/    # 차량 시뮬레이션
│   │   ├── robot/      # 로봇 시뮬레이션
│   │   └── game/       # 게임형 시뮬레이션
│   ├── physics/        # 물리 엔진
│   │   ├── __init__.py
│   │   ├── physics_engine.py        # [구현 필요]
│   │   ├── physics_2d.py            # [구현 필요]
│   │   └── physics_3d.py            # [구현 필요]
│   ├── rendering/      # 렌더링 시스템
│   │   ├── __init__.py
│   │   ├── render_engine.py         # [구현 필요]
│   │   ├── renderer_2d.py           # [구현 필요]
│   │   └── renderer_3d.py           # [구현 필요]
│   ├── ui/             # 사용자 인터페이스
│   │   ├── __init__.py
│   │   ├── main_window.py           # [구현 필요]
│   │   ├── simulation_selector.py   # [구현 필요]
│   │   └── control_panel.py         # [구현 필요]
│   └── config/         # 설정 관리
│       ├── __init__.py
│       ├── config.py                # [구현 필요]
│       └── settings.py              # [구현 필요]
├── tests/              # 테스트 코드
├── examples/           # 예제 프로그램
├── assets/             # 게임 에셋
├── docs/               # 문서
├── main.py             # [구현 필요] 메인 실행 파일
├── pyproject.toml      # [구현 필요] 프로젝트 설정
└── uv.lock             # 의존성 잠금 파일
```

## 기술 스택

### Phase 1-2 (현재 및 다음 단계)

- **Python**: 3.8+
- **물리 엔진**: pymunk (2D)
- **그래픽**: pygame
- **의존성 관리**: uv
- **테스팅**: pytest
- **코드 품질**: ruff
- **문서화**: sphinx (계획)

### Phase 3-4 (향후)

- **3D 물리**: PyBullet
- **3D 그래픽**: ModernGL
- **네트워킹**: asyncio, websockets
- **패키징**: PyInstaller

## 개발 가이드라인

### 코딩 스타일

- PEP 8 준수
- Type hints 사용
- Docstring 작성 (Google 스타일)
- 의미 있는 변수명 사용

### 아키텍처 원칙

- 모듈식 설계
- 추상화 레이어 활용
- 확장 가능한 구조
- 테스트 가능한 코드

### 성능 목표

- 60 FPS 안정성
- 입력 지연 50ms 이하
- 메모리 사용량 1GB 이하
- 시작 시간 5초 이하

## 중요한 파일들

### 문서

- `docs/TRD.md` - 기술 요구사항 명세
- `docs/development-roadmap.md` - 전체 개발 로드맵
- `docs/phase1-foundation.md` - Phase 1 상세 계획
- `docs/phase2-basic-simulations.md` - Phase 2 상세 계획
- `docs/phase3-advanced-features.md` - Phase 3 상세 계획
- `docs/phase4-completion.md` - Phase 4 상세 계획

### 설정 파일

- `pyproject.toml` - 프로젝트 의존성 및 설정
- `uv.lock` - 의존성 잠금 파일
- `.gitignore` - Git 무시 파일

## 다음 작업 우선순위

### 즉시 구현 필요 (Phase 1, Week 1)

1. **pyproject.toml 설정 완성**
   - 의존성 정의 (pygame, pymunk, numpy, etc.)
   - 프로젝트 메타데이터
   - 개발 도구 설정

2. **main.py 기본 구조**
   - 애플리케이션 진입점
   - 기본 초기화 로직

3. **기본 설정 시스템**

   - `pyjoysim/config/config.py`
   - JSON 기반 설정 파일

4. **로깅 시스템**
   - 구조화된 로깅
   - 다양한 로그 레벨

### 이번 주 목표

- [ ] pyproject.toml 완성
- [ ] 기본 실행 가능한 main.py
- [ ] 설정 시스템 기반 구현
- [ ] 로깅 시스템 구현

## 자주 사용하는 명령어

### 개발 환경

```bash
# 의존성 설치
uv sync

# 개발 의존성 포함 설치
uv sync --dev

# 프로젝트 실행
python main.py

# 테스트 실행
pytest

# 코드 포맷팅
ruff format .

# 코드 검사
ruff check .
```

### Git 워크플로우

```bash
# 현재 상태 확인
git status

# 변경사항 커밋
git add .
git commit -m "Add feature: [기능 설명]"

# 브랜치 작업
git checkout -b feature/[기능명]
git checkout main
```

## 개발 참고사항

### 조이스틱 입력 처리

- pygame을 사용한 조이스틱 감지
- 다중 조이스틱 지원 (최대 4개)
- 핫플러그 지원 필요
- 커스텀 키 매핑 기능

### 물리 시뮬레이션

- pymunk를 사용한 2D 물리
- 60 FPS 안정성 중요
- 현실적인 물리 매개변수
- 충돌 감지 최적화

### 렌더링 시스템

- pygame 기반 2D 렌더링
- 확장 가능한 구조 (3D 대비)
- 카메라 시스템 고려
- UI 오버레이 지원

## 테스트 전략

### 단위 테스트

- 각 모듈별 기능 테스트
- Mock 객체 활용
- 90% 코드 커버리지 목표

### 통합 테스트

- 시스템 간 상호작용 검증
- 실제 조이스틱 입력 테스트
- 성능 벤치마킹

### 사용자 테스트

- 다양한 조이스틱 모델 테스트
- 플랫폼별 호환성 확인
- 사용성 평가

## 문제 해결 가이드

### 일반적인 이슈

1. **조이스틱 인식 안됨**
   - 드라이버 확인
   - pygame 초기화 상태 점검

2. **성능 저하**
   - 프로파일링 도구 사용
   - 물리 스텝 크기 조정

3. **의존성 문제**
   - uv sync 재실행
   - 가상환경 재생성

### 디버깅 팁

- 로그 레벨을 DEBUG로 설정
- 단계별 디버깅 활용
- 성능 모니터링 도구 사용

## 추가 메모리

- 파이썬 패키지 관리는 uv를 이용한다.

---

이 문서는 Claude Code 세션에서 프로젝트 컨텍스트를 빠르게 파악할 수 있도록 작성되었습니다. 개발 진행에 따라 지속적으로 업데이트됩니다.