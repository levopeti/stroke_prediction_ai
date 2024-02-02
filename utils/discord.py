import json
from discordwebhook import Discord


class DiscordBot(object):
    def __init__(self, active: bool = True):
        self.active = active

        if active:
            with open('./discord_url.json') as f:
                url = json.load(f)["discord_url"]
            self.discord = Discord(url=url)

    def send_message(self, title=None, description=None, fields=None, file_names=None):
        if self.active:
            if fields is None:
                fields = list()

            if file_names is None:
                file_names = list()
            try:
                files = {filename: open(filename, "rb") for filename in file_names}
                self.discord.post(
                    embeds=[{"title": title, "description": description, "fields": fields}],
                    file=files
                )
            except Exception as e:
                print("training", "error during sent discord message: {}".format(e))
                return
        else:
            print(fields[0]["value"])

