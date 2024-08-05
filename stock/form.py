from django import forms
from .models import StockComment

class CommentForm(forms.ModelForm):
    class Meta:
        model = StockComment
        fields = ['author', 'comment']
        widgets = {
            'comment': forms.Textarea(attrs={'rows': 4}),
        }