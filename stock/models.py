from django.db import models
from member.models import Member


class StockComment(models.Model):
    id = models.AutoField(primary_key=True)  # 기본적으로 AutoField로 자동 증가
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

class RecentStock(models.Model):
    member = models.ForeignKey(Member, on_delete=models.CASCADE)
    stock_code = models.CharField(max_length=10)
    viewed_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-viewed_at']
        unique_together = ('member', 'stock_code')  # 동일한 사용자가 같은 종목을 중복 기록하지 않도록

class InterestStock(models.Model):
    member = models.ForeignKey(Member, on_delete=models.CASCADE)
    stock_code = models.CharField(max_length=10)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ('member', 'stock_code')  # 동일한 사용자가 같은 종목을 중복 기록하지 않도록