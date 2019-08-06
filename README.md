### 1. 환경 만들기

- conda create -n rl python=3.6
- conda activate rl
- pip install --ignore-installed pip
- pip install -r requirements.txt

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