from datetime import date, timedelta

def get_default_dates(fecha_inicio=None, fecha_fin=None):
    if not fecha_inicio or not fecha_fin:
        ayer = date.today() - timedelta(days=1)
        return ayer, ayer
    return fecha_inicio, fecha_fin
