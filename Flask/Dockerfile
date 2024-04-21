FROM python:3.9.11

RUN apt-get update

WORKDIR /usr/src

COPY ./app /usr/src/apps
COPY ./requirements.txt /usr/src/requirements.txt
COPY ./model.pt /usr/src/model.pt

RUN pip install --upgrade pip

RUN echo "building..."

ENV FLASK_APP "apps.app:create_app('local')"
ENV IMAGE_URL "storage/images/"

EXPOSE 5000

CMD ["flask", "run", "-h", "0.0.0.0"]