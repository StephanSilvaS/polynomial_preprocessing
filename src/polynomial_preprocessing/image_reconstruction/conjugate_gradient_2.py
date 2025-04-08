import numpy as np
from scipy import optimize
from math import pi

class ConjugateGradientReconstructor:
    def __init__(self, Vo, Wo, iterations):
        self.Vo = Vo  # Observación en el espacio de Fourier
        self.Wo = Wo  # Máscara (pesos)
        self.n = iterations  # Número de iteraciones
        self.Im = np.zeros_like(Vo, dtype=float)  # Imagen inicial
        
    @staticmethod
    def gauss(ini, dim):
        array_x = np.linspace(-ini, ini, dim)
        array_x = np.reshape(array_x, (dim, 1))
        array_y = np.reshape(array_x, (1, dim))
        img = np.exp(-pi * (array_x**2 + array_y**2))**2
        return img

    @staticmethod
    def norm(weights, x):
        return np.abs(np.sqrt(np.sum(weights * np.abs(x)**2)))

    def f_alpha(self, x, Im, s):
        Vm2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(Im + np.real(x) * s)))
        return np.sum(self.Wo * np.abs(self.Vo - Vm2)**2)

    def reconstruct(self):
        Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im)))
        grad = -np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo * (self.Vo - Vm))))
        s = -grad
        grad_old = np.array(grad)

        for ite in range(self.n):
            diff = -grad
            diff_old = -grad_old

            beta = -np.conjugate(diff) * (diff - diff_old) / np.sum((diff_old * diff_old))
            beta[np.isinf(beta)] = 0
            beta[np.isnan(beta)] = 0

            if ite == 0:
                s = diff
            else:
                s = diff + beta * s

            a = optimize.brent(self.f_alpha, args=(self.Im, s))

            self.Im += a * s

            # Ajustes por restricciones (recortes negativos)
            self.Im.imag[self.Im.real < 0] = np.trunc(self.Im.imag[self.Im.real < 0])
            self.Im.real[self.Im.real < 0] = np.trunc(self.Im.real[self.Im.real < 0])

            grad_old = np.array(grad)
            Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im)))
            grad = -np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo * (self.Vo - Vm))))
            grad[np.isinf(grad)] = 0
            grad[np.isnan(grad)] = 0

        return self.Im
