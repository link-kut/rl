### 1. 환경 만들기

- conda create -n rl python=3.6
- conda activate rl
- pip install --upgrade pip
- pip install -r requirements.txt
- pytorch 설치
  - https://pytorch.org/ 참고
- baselines 설치
  - https://github.com/openai/baselines 참고

### 2. OpenAI Gym 설치

- git clone https://github.com/openai/gym.git
- cd gym
- pip install -e '.[all]'
  - mujoco 에러 무시 
  
  
### 3. 패키지 설치 후 requirements.txt 신규 구성 

- pip freeze > requirements.txt

### 4. mosquitto
- 모스키토 설치
  - brew install mosquitto
  - Linux: https://blog.neonkid.xyz/127

- 모스키토 서비스 실행
  - /usr/local/sbin/mosquitto -c /usr/local/etc/mosquitto/mosquitto.conf
  
- 메세지 구독
  - mosquitto_sub -h [주소] -p [포트] -t [주제]
  - mosquitto_sub -h 127.0.0.1 -p 1883 -t "topic"

- 메시지 발행
  - mosquitto_pub -h [주소] -p [포트] -t [주제] -m [메세지]
  - mosquitto_pub -h 127.0.0.1 -p 1883 -t "topic" -m "test messgae"
  
### 5. 실행
- main.py 실행

  
### 6. gitignore 적용

- git rm -r --cached .
- git add .
- git commit -m "Apply .gitignore"  


### 7. 새로운 패키지 설치 후 requirements.txt 새로 만들기

- pip freeze > requirements.txt

### 8. MuJoCo 설치

- brew install gcc@6
- mkdir ~/.mujoco
- https://github.com/openai/mujoco-py 에서 다운로드
- mv Downloads/mujoco200_macos ~/.mujoco/mujoco200
- cp Downloads/mjkey.txt ~/.mujoco/
- pip install -U 'mujoco-py<2.1,>=2.0'

### 참고 문헌
- https://medium.com/aureliantactics/ppo-hyperparameters-and-ranges-6fc2d29bccbev
- https://arxiv.org/pdf/1709.06009.pdf
- https://medium.com/@jonathan_hui/rl-proximal-policy-optimization-ppo-explained-77f014ec3f12
- https://en.wikipedia.org/wiki/MM_algorithm
- https://drive.google.com/file/d/0BxXI_RttTZAhMVhsNk5VSXU0U3c/view
