# Generated by Django 4.2.7 on 2023-11-11 14:12

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('myapp', '0002_platnomor_delete_gambarmodel'),
    ]

    operations = [
        migrations.AlterModelTable(
            name='platnomor',
            table='detection_platnomor',
        ),
    ]