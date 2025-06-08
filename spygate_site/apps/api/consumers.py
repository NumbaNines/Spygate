import json
from channels.generic.websocket import AsyncWebsocketConsumer

class AnalysisConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        # Get the analysis ID from the URL route
        self.analysis_id = self.scope['url_route']['kwargs']['analysis_id']
        self.room_group_name = f'analysis_{self.analysis_id}'

        # Join the analysis group
        await self.channel_layer.group_add(
            self.room_group_name,
            self.channel_name
        )
        await self.accept()

    async def disconnect(self, close_code):
        # Leave the analysis group
        await self.channel_layer.group_discard(
            self.room_group_name,
            self.channel_name
        )

    async def receive(self, text_data):
        # Handle incoming messages from WebSocket
        text_data_json = json.loads(text_data)
        message_type = text_data_json.get('type')
        data = text_data_json.get('data')

        # Send message to the analysis group
        await self.channel_layer.group_send(
            self.room_group_name,
            {
                'type': 'analysis_update',
                'message_type': message_type,
                'data': data
            }
        )

    async def analysis_update(self, event):
        # Send analysis update to WebSocket
        await self.send(text_data=json.dumps({
            'type': event['message_type'],
            'data': event['data']
        })) 