from setuptools import setup, find_packages

setup(
    name='arclab_mit',
    version='0.1.0',
    description='Scripts for arclab_mit project',
    author='Tu Nombre',
    author_email='tu.email@example.com',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'krpc',
        'Pillow',
        'opencv-python',
        'termcolor',
        'requests',
        'python-dotenv',
        'anthropic',
        'astropy',
    ],
    entry_points={
        'console_scripts': [
            'claude_vision_agent=agents.claude_vision_agent:main',
            'vision_few_shot_LLM_agent=agents.vision_few_shot_LLM_agent:main',
            'vision_LLM_agent=agents.vision_LLM_agent:main',
            'claude_vision_roulette=agents.claude_vision_roulette:main',
            'navball_agent=agents.navball_agent:main',
        ],
    },
)