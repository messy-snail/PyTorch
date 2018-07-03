{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import sys\n",
    "from PyQt5.QtWidgets import *\n",
    "from PyQt5 import uic\n",
    "\n",
    "from_class = uic.loadUiType(\"main_window.ui\")[0]\n",
    "\n",
    "class MyWindow(QMainWindow, from_class):\n",
    "    def __init__(self):\n",
    "        super().__init__()\n",
    "        self.setupUi(self)\n",
    "        self.pushButton.clicked.connect(self.btn_clicked)\n",
    "        \n",
    "    def btn_clicked(self):\n",
    "        QMessageBox.about(self, \"message\", \"clicked\")\n",
    "        \n",
    "if __name__ == \"__main__\":\n",
    "    app = QApplication(sys.argv)\n",
    "    myWindow = MyWindow()\n",
    "    myWindow.show()\n",
    "    app.exec_()\n",
    "    \n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.6.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
