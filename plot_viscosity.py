from matplotlib import pyplot as plt
import torch

n = 1.3
min_fact = 0.3
gamma_scale = 15


def get_viscostiy(shear_rate):
    gamma_dot_sq = shear_rate ** 2
    # Power law
    limit = 1 / min_fact
    power_factor = gamma_scale ** (1 - n) * gamma_dot_sq ** ((n - 1) / 2)
    power_factor = limit * torch.tanh((power_factor / limit) ** 2) ** (1 / 2)
    # Carreau
    carreau_factor = min_fact + (2 - 2 * min_fact) * (1 + 4*gamma_scale ** (-2) * gamma_dot_sq) ** ((n - 1) / 2)
    if n > 1:
        carreau_factor = carreau_factor - 2 + 2 * min_fact

    # Herschel–Bulkley approximate
    s = torch.sigmoid((gamma_scale - gamma_dot_sq.sqrt()) / (0.1 * gamma_scale))
    herschel_factor = min_fact + (2.5 - min_fact) * s
    if n > 1:
        herschel_factor = 1 / herschel_factor

    return power_factor, carreau_factor, herschel_factor


def main():
    shear_rate = torch.linspace(0, 50, 100)
    power, carreau, herschel = get_viscostiy(shear_rate)

    plt.plot(shear_rate.numpy(), power.numpy(), label="Power Law")
    plt.plot(shear_rate.numpy(), carreau.numpy(), label="Carreau")
    plt.plot(shear_rate.numpy(), herschel.numpy(), label="Herschel-Bulkley")
    plt.axhline(y=1, color="black", linestyle="--", label="Newtonian")
    plt.xlabel("Shear Rate")
    plt.ylabel("Viscosity Factor")
    plt.title("Viscosity Models")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()