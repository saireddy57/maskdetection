FROM python:3.8.10
RUN apt-get update && apt-get install -y libgl1-mesa-glx
COPY . /app
WORKDIR /app
# Upgrade pip
# RUN sudo apt-get update
# RUN sudo apt-get install libgl1-mesa-glx
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install -r requirements.txt
ENTRYPOINT [ "python" ]
CMD [ "main.py" ]
