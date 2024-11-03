from setuptools import find_packages, setup

package_name = 'isaac_sim_ai'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kabilankb',
    maintainer_email='kabilankb@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
                 'segmentation = isaac_sim_ai.segmentation:main',
                 'isaacsim_human_following = isaac_sim_ai.isaacsim_human_following:main',
                 'isaacsim_humanpose = isaac_sim_ai.isaacsim_humanpose:main',
                 'isaacsim_monodepth = isaac_sim_ai.isaacsim_monodepth:main',
                 'isaacsim_objecct_detection = isaac_sim_ai.isaacsim_objecct_detection:main',
                 'isaacsim_object_avoidance = isaac_sim_ai.isaacsim_object_avoidance:main',
                 'isaacsim_jetbot_balltracking = isaac_sim_ai.isaacsim_jetbot_balltracking:main',
                 'isaacsim_jetbot_linefollowing = isaac_sim_ai.isaacsim_jetbot_linefollowing:main',
                 'isaacsim_kaya = isaac_sim_ai.isaacsim_kaya:main',
                 'isaacsim_kaya_greencube = isaac_sim_ai.isaacsim_kaya_greencube:main',
        ],
    },
)
