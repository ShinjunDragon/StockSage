# Generated by Django 5.0.7 on 2024-08-06 01:10

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('stock', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='comment',
            name='member_id',
            field=models.CharField(max_length=20),
        ),
    ]
