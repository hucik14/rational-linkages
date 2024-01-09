from matplotlib.widgets import Button


class ButtonID(Button):
    def __init__(
        self, ax, label, image=None, color="0.85", hovercolor="0.95", button_id=1
    ):
        """
        Button class derivative to support an ID assignment to a button. ID then
        corresponds a joint order.

        :param ax:
        :param label:
        :param image:
        :param color:
        :param hovercolor:
        :param button_id:
        """
        Button.__init__(self, ax, label, image, color, hovercolor)
        self.func = None
        self.button_id = button_id

    def on_clicked(self, func):
        """
        Method override for "on_clicked" function. func is saved and handled later
        in "_clicked" method together
        with the ID.

        :param func: original func
        :return:
        """
        Button.on_clicked(self, self._clicked)
        self.func = func

    def _clicked(self, event):
        """
        Method that adds ID to the func values.

        :param event:
        :return:
        """
        self.func(event, self.button_id)
