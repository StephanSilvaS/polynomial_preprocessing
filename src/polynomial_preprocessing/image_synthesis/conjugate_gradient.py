import numpy as np
from math import e, pi
from array import *
from scipy import optimize
from scipy.signal import convolve2d as conv

class ConjugateGradient:
    def __init__(self, Vo, Wo, n):
        self.Vo = Vo
        self.Wo = Wo
        self.n = n
        self.Im = np.zeros_like(Vo, dtype=float)
        self.Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im)))
        self.grad = -1 * np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(Wo * (Vo - self.Vm))))
        self.grad_old = np.array(self.grad)
        self.s = -self.grad

    def gauss(self, ini, dim):
        array_x = np.linspace(-ini, ini, dim)
        array_x = np.reshape(array_x, (dim, 1))
        array_y = np.reshape(array_x, (1, dim))
        img = np.exp(-pi * (array_x**2 + array_y**2))**2
        return img

    def f_alpha(self, x: float, s):
        Vm2 = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im + np.real(x) * s)))
        return np.sum(self.Wo * np.absolute(self.Vo - Vm2)**2)

    def compute_gradient(self):
        for ite in range(self.n):
            diff = -self.grad
            diff_old = -self.grad_old

            if ite == 0:
                self.s = diff
            else:
                # Cálculo de beta sin el signo negativo extra y usando producto escalar real
                beta = np.sum(diff * (diff - diff_old)) / np.sum(diff_old * diff_old)
                self.s = diff + beta * self.s

            # Búsqueda del parámetro óptimo de escala a
            a = optimize.brent(self.f_alpha, args=(self.s,))
            
            # Actualizamos la imagen y forzamos su parte real
            self.Im = np.real(self.Im + a * self.s)

            self.grad_old = np.array(self.grad)
            self.Vm = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(self.Im)))
            self.grad = -np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(self.Wo * (self.Vo - self.Vm))))
            self.grad[np.isinf(self.grad)] = 0
            self.grad[np.isnan(self.grad)] = 0
            
            # (Opcional) Puedes evaluar la norma del gradiente para decidir si detener el ciclo
            # if np.linalg.norm(self.grad) < tol:
            #     break

        return self.Im


    def norm(self, weights, x):
        return np.absolute(np.sqrt(np.sum(weights * np.absolute(x)**2)))