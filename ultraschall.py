"""HC-SR04 an TRIG=GPIO23 (Pin 16), ECHO=GPIO24 (Pin 18).
Hinweis: ECHO ist ohne Spannungsteiler direkt angeschlossen (5V an 3V3-Pin).
Bei sporadischen Fehlmessungen zuerst hier nachsehen. Ausweichpin: GPIO25."""
import lgpio, time, statistics, threading

TRIG, ECHO = 23, 24

class Ultraschall:
    def __init__(self):
        self.h = lgpio.gpiochip_open(0)
        lgpio.gpio_claim_output(self.h, TRIG, 0)
        lgpio.gpio_claim_input(self.h, ECHO)
        self.lock = threading.Lock()

    def _puls(self, timeout=0.06):
        lgpio.gpio_write(self.h, TRIG, 0); time.sleep(0.002)
        lgpio.gpio_write(self.h, TRIG, 1); time.sleep(0.00001)
        lgpio.gpio_write(self.h, TRIG, 0)

        t0 = time.perf_counter()
        while lgpio.gpio_read(self.h, ECHO) == 0:
            if time.perf_counter() - t0 > timeout:
                return None
        start = time.perf_counter()
        while lgpio.gpio_read(self.h, ECHO) == 1:
            if time.perf_counter() - start > timeout:
                return None
        return (time.perf_counter() - start) * 34300 / 2

    def messen(self, n=5):
        with self.lock:
            werte = []
            for _ in range(n):
                v = self._puls()
                if v is not None:
                    werte.append(v)
                time.sleep(0.07)          # <60ms erzeugt Nachhall-Fehler
        return werte
