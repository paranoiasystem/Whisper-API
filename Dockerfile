FROM python:3.10-slim

WORKDIR /whisper_api

RUN apt-get update && apt-get install git -y
RUN pip3 install flask
RUN pip3 install "git+https://github.com/openai/whisper.git" 
RUN apt-get install -y ffmpeg

COPY . .

EXPOSE 5000

CMD [ "python3", "-m" , "flask", "run", "--host=0.0.0.0"]