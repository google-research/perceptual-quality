import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="perceptual-quality",
    version="0.0.1",
    author="Sangnie Bhardwaj, Johannes Ballé and Ian Fischer",
    description="A small example package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/google-research/perceptual-quality",
    packages=setuptools.find_packages(),
)