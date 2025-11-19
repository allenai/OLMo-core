import matplotlib.pyplot as plt

# plt.style.use("seaborn-white")

fig, ax = plt.subplots(figsize=(5, 20))


fontweights = ["normal", "bold", "heavy"]
fontsize = 30

for i, fw in enumerate(fontweights):
    ax.text(
        0.1,
        (len(fontweights) - i) / len(fontweights),
        f"fontweight: {fw}",
        weight=fw,
        fontsize=fontsize,
        fontfamily="Manrope",
        transform=ax.transAxes,
    )

ax.set_axis_off()

plt.show()
