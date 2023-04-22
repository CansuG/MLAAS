FROM python:3

WORKDIR /MLAAS_API

ADD . /MLAAS_API

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "app.py" ]