from django.db import models

class PlatNomor(models.Model):
    nomor = models.CharField(max_length=20)
    gambar = models.ImageField(upload_to='platnomor/')
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.nomor
    
    class Meta:
        db_table = 'detection_platnomor'
