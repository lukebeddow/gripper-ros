#!/bin/bash

# use xacro to prepare the xml files

echo making objects.xml
xacro objects.xacro > objects.xml

echo making assets.xml
xacro assets.xacro > assets.xml

echo making details.xml
xacro details.xacro > details.xml