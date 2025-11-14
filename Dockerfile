FROM python:3.13.7

RUN mkdir -p laptop_price_prediction

WORKDIR /laptop_price_prediction

COPY . /laptop_price_prediction

RUN pip install -r requirements.txt

ENV PORT=8000

EXPOSE 8000

CMD ["uvicorn","app:app","--host","0.0.0.0","--port","8000"]