#
# Imported module functions
#
#https://camo.githubusercontent.com/582226b9ba41bcbc13eaa81d2764092abb443bd416578c175bc2c1c5742d0647/68747470733a2f2f692e696d6775722e636f6d2f6b7a6978316a492e706e67
# Use our SimpleRequests module for this experimental version.
from SimpleRequests import SimpleRequest
from SimpleRequests.SimpleRequest import error

# Use the datetime module for generating timestamps and snowflakes.
from datetime import datetime, timedelta,timezone

# Use the time module for generating timestamps that are backwards compatible with Python 2.
from time import mktime

# Use the os module for creating directories and writing files.
from os import makedirs, getcwd, path

# Use the mimetypes module to determine the mimetype of a file.
from mimetypes import MimeTypes

# Use the sqlite3 module to access SQLite databases.
from sqlite3 import connect, Row, IntegrityError

# Use the random module to choose from a list at random.
from random import choice

# Convert JSON to a Python dictionary for ease of traversal.
from json import loads

import dateutil.parser

import textmine as tx

from concurrent.futures import ThreadPoolExecutor as pool

import logging


import asyncio
from contextlib import suppress

#
# Lambda functions
#
logging.basicConfig(filename='./output.log', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)

# Return a random string of a specified length.
random_str = lambda length: ''.join([choice('0123456789ABCDEF') for i in range(length)])

# Get the mimetype string from an input filename.
mimetype = lambda name: MimeTypes().guess_type(name)[0] \
    if MimeTypes().guess_type(name)[0] is not None \
    else 'application/octet-stream'

# Return a Discord snowflake from a timestamp.
snowflake = lambda timestamp_s: (timestamp_s * 1000 - 1420070400000) << 22

# Return a timestamp from a Discord snowflake.
timestamp = lambda snowflake_t: ((snowflake_t >> 22) + 1420070400000) / 1000.0

time_dff = 4
#
# Global functions
#

class Message(object):
    def __init__(self, id, user_id, timestamp, content):
        self.id = id
        self.user_id = user_id
        self.timestamp = timestamp
        self.content = content


def snowtodatetime(snowflake_value):
    ts = ((snowflake_value / 4194304) + 1420070400000)/1000
    timestamp = datetime.utcfromtimestamp(ts)
    return timestamp

def utctosnow(timestamp):
    return((timestamp*1000) - 1420070400000) * 4194304
    


def get_day(day, month, year):
    """Get the timestamps from 00:00 to 23:59 of the given day.

    :param day: The target day.
    :param month: The target month.
    :param year: The target year.
    """

    min_time = mktime((year, month, day, 0, 0, 0, -1, -1, -1))
    max_time = mktime((year, month, day, 23, 59, 59, -1, -1, -1))


    return {
        '00:00': snowflake(int(min_time)),
        '23:59': snowflake(int(max_time))
    }


def safe_name(name):
    """Convert name to a *nix/Windows compliant name.

    :param name: The filename to convert.
    """

    output = ""
    for char in name:
        if char not in '\\/<>:"|?*':
            output += char

    return output


def create_query_body(**kwargs):
    """Generate a search query string for Discord."""

    query = ""

    for key, value in kwargs.items():
        if value is True and key != 'nsfw':
            query += '&has=%s' % key[:-1]

        if key == 'nsfw':
            query += '&include_nsfw=%s' % str(value).lower()

    return query



class DiscordConfig(object):
    """Just a class used to store configs as objects."""


class Discord:
    """Experimental Discord scraper class."""

    def __init__(self, config='config.json', apiver='v6'):
        """Discord constructor.

        :param config: The configuration JSON file.
        :param apiver: The current Discord API version.
        """


        with open(config, 'r') as configfile:
            configdata = loads(configfile.read())

        cfg = type('DiscordConfig', (object,), configdata)()
        if cfg.token == "" or cfg.token is None:
            error('You must have an authorization token set in %s' % config)
            exit(-1)

        self.api = apiver
        self.buffer = cfg.buffer

        self.headers = {
            'user-agent': cfg.agent,
            'authorization': cfg.token
        }

        self.types = cfg.types
        self.query = create_query_body(
            images=cfg.query['images'],
            files=cfg.query['files'],
            embeds=cfg.query['embeds'],
            links=cfg.query['links'],
            videos=cfg.query['videos'],
            nsfw=cfg.query['nsfw']
        )

        self.directs = cfg.directs if len(cfg.directs) > 0 else {}
        self.servers = cfg.servers if len(cfg.servers) > 0 else {}

        # Save us the time by exiting out when there's nothing to scrape.
        if len(cfg.directs) == 0 and len(cfg.servers) == 0:
            error('No servers or DMs were set to be grabbed, exiting.')
            exit(0)

        '''
        dbdir = path.join(getcwd(), 'data')
        if not path.exists(dbdir):
            makedirs(dbdir)

        dbfile = path.join(dbdir, 'users.db')
        self.db = connect(dbfile)
        self.c = self.db.cursor()
        self.c.row_factory = Row
        '''

        self.tx_obj = tx.NLPstock()

        self.start_time = None
        self.end_time = None

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)




    def get_server_name(self, serverid, isdm=False):
        """Get the server name by its ID.

        :param serverid: The server ID.
        :param isdm: A flag to check whether we're in a DM or not.
        """

        if isdm:
            return serverid

        request = SimpleRequest(self.headers).request
        server = request.grab_page('https://discordapp.com/api/%s/guilds/%s' % (self.api, serverid))

        if server is not None and len(server) > 0:
            return '%s_%s' % (serverid, safe_name(server['name']))

        else:
            error('Unable to fetch server name from id, generating one instead.')
            return '%s_%s' % (serverid, random_str(12))

    def get_channel_name(self, channelid, isdm=False):
        """Get the channel name by its ID.

        :param channelid: The channel ID.
        :param isdm: A flag to check whether we're in a DM or not.
        """

        if isdm:
            return channelid

        request = SimpleRequest(self.headers).request
        channel = request.grab_page('https://discordapp.com/api/%s/channels/%s' % (self.api, channelid))

        if channel is not None and len(channel) > 0:
            return '%s_%s' % (channelid, safe_name(channel['name']))

        else:
            error('Unable to fetch channel name from id, generating one instead.')
            return '%s_%s' % (channelid, random_str(12))

    @staticmethod
    def create_folders(server, channel):
        """Create the folder structure.

        :param server: The server name.
        :param channel: The channel name.
        """

        folder = path.join(getcwd(), 'data', server, channel)
        if not path.exists(folder):
            makedirs(folder)

        return folder

    def download(self, url, folder):
        """Download the contents of a URL.

        :param url: The target URL.
        :param folder: The target folder.
        """

        request = SimpleRequest(self.headers).request
        request.set_header('user-agent', 'Mozilla/5.0 (X11; Linux x86_64) Chrome/78.0.3904.87 Safari/537.36')

        filename = safe_name('%s_%s' % (url.split('/')[-2], url.split('/')[-1]))
        if not path.exists(filename):
            request.stream_file(url, folder, filename, self.buffer)

    def check_config_mimetypes(self, source, folder):
        """Check the config settings against the source mimetype.

        :param source: Response from Discord search.
        :param folder: Folder where the data will be stored.
        """

        for attachment in source['attachments']:
            if self.types['images'] is True:
                if mimetype(attachment['proxy_url']).split('/')[0] == 'image':
                    self.download(attachment['proxy_url'], folder)

            if self.types['videos'] is True:
                if mimetype(attachment['proxy_url']).split('/')[0] == 'video':
                    self.download(attachment['proxy_url'], folder)

            if self.types['files'] is True:
                if mimetype(attachment['proxy_url']).split('/')[0] not in ['image', 'video']:
                    self.download(attachment['proxy_url'], folder)

    @staticmethod
    def insert_text(server, channel, message):
        """Insert the text data into our SQLite database file.

        :param server: The server name.
        :param channel: The channel name.
        :param message: Our message object.
        """

        dbdir = path.join(getcwd(), 'data')
        if not path.exists(dbdir):
            makedirs(dbdir)

        dbfile = path.join(dbdir, 'text.db')
        db = connect(dbfile)
        c = db.cursor()
        

        c.execute('''CREATE TABLE IF NOT EXISTS text_%s_%s (
            id TEXT,
            name TEXT,
            content TEXT,
            timestamp TEXT
        )''' % (server, channel))

        c.execute('INSERT INTO text_%s_%s VALUES (?,?,?,?)' % (server, channel), (
            message['author']['id'],
            '%s#%s' % (message['author']['username'], message['author']['discriminator']),
            message['content'],
            message['timestamp']
        ))

        #print(message.keys())
        #print(f"{message['author']['id']} {message['author']['username']} {message['author']['discriminator']} {message['timestamp']}")
        #dt_time = dateutil.parser.isoparse(message['timestamp'])
        #ts_comp = dt_time.replace(tzinfo=timezone.utc).timestamp()


        print(f"{message['content']} {message['timestamp']}")

        db.commit()
        db.close()

    def check_AH(self, dt):
        start = dt.replace(hour=9, minute=30, second=0, microsecond=0)
        end = dt.replace(hour=16, minute=0, second=0, microsecond=0)
        if dt > start:
            if dt > end:
                return True
            else:
                return False
        else:
            return True


    def insert_text_player(self, server, channel, message, message_hour):
        """Insert the text data into our SQLite database file.

        :param server: The server name.
        :param channel: The channel name.
        :param message: Our message object.
        """
        global time_dff

        dbdir = path.join(getcwd(), 'data')
        if not path.exists(dbdir):
            makedirs(dbdir)

        dbfile = path.join(dbdir, 'user.db')
        db = connect(dbfile)
        c = db.cursor()

        '''
        if self.check_AH(message_hour+timedelta(hours= -time_dff)):
            self.tx_obj.AH = True
            logging.info(f"staring after hours for the day {message_hour+timedelta(hours= -time_dff)}")
        else:
            self.tx_obj.AH = False
        '''
        self.tx_obj.current_time = message_hour

        #try:
        stock_string = self.tx_obj.get_stocks(message)
        #except Exception as e:
            #logging.error(f"getting stocks error {e} {message}")

        


        mentions = message["mentions"]
        if mentions:
            try:
                reference = message['message_reference']

                try:
                    c.execute("SELECT * FROM text_%s_%s WHERE id = ?" % (server, mentions[0]['id']) , (reference['message_id'],)) 
                    #rows = self.c.fetchall()
                    #mention_stock_string = rows[-1]
                    #print("EXECUTING finding message from refered user: ", mentions[0]['id'])
                except Exception as e:
                    #print("cant find token table from user ", mentions[0]['id'])
                    pass


            except KeyError:
                #print("not reply simply pin acess last topics org")
                try:
                    c.execute('SELECT * FROM text_%s_%s ORDER BY id DESC LIMIT 1' % (server, mentions[0]['id']))
                    #print("EXECUTING finding last message from pinned user: ", mentions[0]['id'])
                except Exception:
                    pass



            result = c.fetchone()
            if result:
                #print(f"ORG from {mentions[0]['id']} is {result[-1]} {result[2]}")
                stocks_temp = result[-1].split()
                stock_string += stocks_temp
                stock_string = set(stock_string)



           
            

            #stock_string += mention_stock_string

        stock_string = ' '.join(stock_string)

        c.execute('''CREATE TABLE IF NOT EXISTS text_%s_%s (
            id TEXT NOT NULL PRIMARY KEY,
            name TEXT,
            content TEXT,
            timestamp TEXT,
            stocks TEXT
        )''' % (server, message['author']['id']))

        c.execute('INSERT INTO text_%s_%s VALUES (?,?,?,?,?)' % (server, message['author']['id']), (
            message['id'],
            channel,
            message['content'],
            message['timestamp'],
            stock_string
        ))

        #print(message.keys())
        #print(f"{message['author']['id']} {message['author']['username']} {message['author']['discriminator']} {message['timestamp']}")
        #dt_time = dateutil.parser.isoparse(message['timestamp'])
        #ts_comp = dt_time.replace(tzinfo=timezone.utc).timestamp()
        print(f"{message['content']} - stocks: {stock_string}")
  

        db.commit()
        db.close()
        




    def grab_data_test(self, folder, server, channel, isdm=False, inter=30):
        """Scan and grab the attachments.

        :param folder: The folder name.
        :param server: The server name.
        :param channel: The channel name.
        :param isdm: A flag to check whether we're in a DM or not.
        :param inter: interval of scrape in seconds
        """


        date = datetime.now()
        target_day = date + timedelta(days=-200)
        
        while target_day.day <= date.day:
            print(f"getting data for {date} target is {target_day}")
            
            #start_snow = int(utctosnow(date.replace(day = date.day-1, hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc).timestamp()))
            #end_snow = int(utctosnow(date.replace(hour=23, minute=59, second=59, microsecond=59, tzinfo=timezone.utc).timestamp()))
            today = get_day(target_day.day, target_day.month, target_day.year)

            start_snow = today["00:00"]
            end_snow = today['23:59']
            print(f"{start_snow}-{end_snow}")
            print()

            request = SimpleRequest(self.headers).request

            
            request.set_header('referer', 'https://discordapp.com/channels/@me/%s' % channel)
            content = request.grab_page(
                'https://discordapp.com/api/%s/channels/%s/messages/search?min_id=%s&max_id=%s&%s' %
                (self.api, channel, start_snow, end_snow, self.query)
            )

            try:
                if content['messages'] is not None:
                    for messages in content['messages'][::-1]:
                        for message in messages[::-1]:
                            #self.check_config_mimetypes(message, folder)

                            if self.types['text']:
                                if len(message['content']) > 0:
                                    try:
                                        self.insert_text_player(server, channel, message)
                                    except IntegrityError:
                                        pass
            except TypeError as e:
                print("type error on getting message ", e)

            #break

            target_day += timedelta(days=1)

    def grab_server_data(self):
        """Scan and grab the attachments within a server."""

        for server, channels in self.servers.items():
            for channel in channels:
                folder = self.create_folders(
                    self.get_server_name(server),
                    self.get_channel_name(channel)
                )

                self.grab_data_current(folder, server, channel)

    def grab_dm_data(self):
        """Scan and grab the attachments within a direct message."""

        for alias, channel in self.directs.items():
            folder = self.create_folders(
                path.join('Direct Messages', alias),
                channel
            )

            self.grab_data(folder, alias, channel, True)


        



    async def grab_data_current(self, server, channel, isdm=False, inter=30):

        #the end time
        """Scan and grab the attachments.

        :param folder: The folder name.
        :param server: The server name.
        :param channel: The channel name.
        :param isdm: A flag to check whether we're in a DM or not.
        :param inter: interval of scrape in seconds
        """


        global time_dff


        inter_before = datetime.now() + timedelta(hours=time_dff)
        print("current time is ", inter_before)
        inter_after = inter_before + timedelta(seconds=inter)
        #ts_value_now = dt_time.replace(tzinfo=timezone.utc).timestamp()

        while True:


            current_time = datetime.now() + timedelta(hours=time_dff)
            #print(f"waiting for {inter_after}, current {current_time}")
            if current_time >= inter_after:



                #inter_before -= timedelta(seconds=5) #offset to get the overlap message
                start_snow_dt = inter_before.replace(tzinfo=timezone.utc) + timedelta(seconds=-2)
                start_snow = int(utctosnow(start_snow_dt.timestamp()))
                end_snow_dt = inter_after.replace(tzinfo=timezone.utc) + timedelta(seconds=2)
                end_snow = int(utctosnow(end_snow_dt.timestamp()))

                print(f"Processing time interval {inter_before} to {current_time}")



                request = SimpleRequest(self.headers).request
                
                request.set_header('referer', 'https://discordapp.com/channels/%s/%s' % (server, channel))
                content = request.grab_page(
                    'https://discordapp.com/api/%s/guilds/%s/messages/search?channel_id=%s&min_id=%s&max_id=%s&%s' %
                    (self.api, server, channel, start_snow, end_snow, self.query)
                )
                

                if content:
                    if content['messages'] is not None:
                        for messages in content['messages'][::-1]:
                            for message in messages[::-1]:
                                


                                #self.check_config_mimetypes(message, folder)
                                #print(message['id'])
                                if self.types['text'] is True:
                                    if len(message['content']) > 0:
                                        try:
                                            self.insert_text_player(server, channel, message, start_snow_dt)
                                        except IntegrityError:
                                            logging.error(f"{message['id']} exists by {message['author']['id']} {message['content']} {message['author']['username']}")
                else:
                    logging.info(f"{start_snow_dt}-{end_snow_dt} no content {content}")

                        




            

                inter_before = current_time
                inter_after = inter_before + timedelta(seconds=inter)

                print()
            await asyncio.sleep(0.5)


    def grab_data(self, folder, server, channel, isdm=False):
        """Scan and grab the attachments.

        :param folder: The folder name.
        :param server: The server name.
        :param channel: The channel name.
        :param isdm: A flag to check whether we're in a DM or not.
        """

        date = datetime.today()

        while date.year >= 2021:
            request = SimpleRequest(self.headers).request
            today = get_day(date.day, date.month, date.year)

            if not isdm:
                request.set_header('referer', 'https://discordapp.com/channels/%s/%s' % (server, channel))
                content = request.grab_page(
                    'https://discordapp.com/api/%s/guilds/%s/messages/search?channel_id=%s&min_id=%s&max_id=%s&%s' %
                    (self.api, server, channel, today['00:00'], today['23:59'], self.query)
                )
            else:
                request.set_header('referer', 'https://discordapp.com/channels/@me/%s' % channel)
                content = request.grab_page(
                    'https://discordapp.com/api/%s/channels/%s/messages/search?min_id=%s&max_id=%s&%s' %
                    (self.api, channel, today['00:00'], today['23:59'], self.query)
                )

            try:
                if content['messages'] is not None:
                    for messages in content['messages']:
                        for message in messages:
                            #self.check_config_mimetypes(message, folder)

                            if self.types['text'] is True:
                                if len(message['content']) > 0:
                                    self.insert_text(server, channel, message)
            except TypeError:
                continue
            break
            date += timedelta(days=-1)

    def grab_server_data(self):
        """Scan and grab the attachments within a server."""

        for server, channels in self.servers.items():
            for channel in channels:
                print(f'Scraping data from {self.get_server_name(server)} {self.get_channel_name(channel)}')

                self.loop.create_task(self.grab_data_current(server, channel))

        self.loop.run_forever()
                


    def grab_dm_data(self):
        """Scan and grab the attachments within a direct message."""

        for alias, channel in self.directs.items():
            folder = self.create_folders(
                path.join('Direct Messages', alias),
                channel
            )

            self.grab_data(folder, alias, channel, True)





#
# Initializer
#
if __name__ == '__main__':
    ds = Discord()
    ds.grab_server_data()

    #ds.grab_dm_data()

    
    