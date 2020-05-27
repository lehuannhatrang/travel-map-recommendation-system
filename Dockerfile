FROM python:3.5.6-alpine

WORKDIR /usr/src/app

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

RUN apk upgrade --update
RUN apk add --no-cache gcc musl-dev libffi-dev libxml2-dev libxslt-dev openssl-dev
RUN pip install --upgrade pip

COPY ./requirements.txt /usr/src/app/requirements.txt

RUN apk add --no-cache postgresql-libs && \
    apk add --no-cache --virtual .build-deps gcc musl-dev postgresql-dev && \
    python3 -m pip install -r requirements.txt --no-cache-dir && \
    apk --purge del .build-deps

COPY ./redis.conf /usr/src/app/redis.conf

#RUN pip install -r requirements.txt

# copy project
COPY . /usr/src/app/
EXPOSE 8086