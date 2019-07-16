### 0. 환경 만들기 및 모듈 설치

- conda create -n rl python=3.7
- conda activate rl
- pip install --ignore-installed pip

### 1. OpenAI Gym 설치

- git clone https://github.com/openai/gym.git
- cd gym
- pip install -e '.[all]'
  - mujoco 에러 무시 

### 2. mosquitto
- 모스키토 설치
  - brew install mosquitto

- 모스키토 서비스 실행
  - /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
  
- 메세지 구독
  - mosquitto_sub -h [주소] -p [포트] -t [주제]
  - mosquitto_sub -h 127.0.0.1 -p 1883 -t "topic"

- 메시지 발행
  - mosquitto_pub -h [주소] -p [포트] -t [주제] -m [메세지]
  - mosquitto_pub -h 127.0.0.1 -p 1883 -t "topic" -m "test messgae"
  
### 3. constants.py 내 설정 내용 조정
- constants.py 파일 열기
  - 적절한 anaconda environment 확인
  - PYTHON_PATH="~/anaconda3/envs/rl/bin/python3"의 Python 인터프리터 패스 수정
  
### 4. 실행
- main.py 실행

### 5. Cartpole-0
- https://github.com/openai/gym/wiki/CartPole-v0

- Action
  - 0: Left
  - 1: Right
  
- Reward
  - Reward is 1 for every step taken, including the termination step

- Starting State
  - All observations are assigned a uniform random value between ±0.05

- Episode Termination
  - Pole Angle is more than ±12°
  - Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
  - Episode length is greater than 200

- Solved Requirements
  - Considered solved when the average reward is greater than or equal to 195.0 over 100 consecutive trials.
  
### gitignore 적용

- git rm -r --cached .
- git add .
- git commit -m "Apply .gitignore"  