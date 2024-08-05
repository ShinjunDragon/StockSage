
from django.db import models

class StockComment(models.Model):
    ticker = models.CharField(max_length=10)
    author = models.CharField(max_length=100)
    comment = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.author} - {self.ticker}"

'''
    만들고 난 뒤
    python manage.py migrate
    python manage.py makemigrations
    python manage.py migrate
    순서대로 해서 StockComment 테이블 생성해야 함
'''