import os
import requests
from datetime import datetime, timedelta
from langchain_core.tools import tool
import re
import dns.resolver # You'll need to add 'dnspython' to requirements.txt

# The primary calendar where the event is created
TARGET_CALENDAR = "GoDataConsultations@godata.com.ec"


def get_ms_graph_token():
    tenant_id = os.getenv("MS_TENANT_ID")
    client_id = os.getenv("MS_CLIENT_ID")
    client_secret = os.getenv("MS_CLIENT_SECRET")
    
    url = f"https://login.microsoftonline.com/{tenant_id}/oauth2/v2.0/token"
    payload = {
        "client_id": client_id,
        "scope": "https://graph.microsoft.com/.default",
        "client_secret": client_secret,
        "grant_type": "client_credentials"
    }
    
    response = requests.post(url, data=payload)
    response.raise_for_status()
    return response.json().get("access_token")

@tool
def book_godata_meeting(user_name: str, user_email: str, company_name: str, date: str, time: str) -> str:
    """
    Use this tool ONLY when you have all 5 pieces of information: Name, Email, Company, Date (YYYY-MM-DD), and Time (HH:MM in 24-hour format).
    """
    print(f"✅ ATTEMPTING GRAPH API BOOKING: {user_name} on {date} at {time}")
    
    try:
        token = get_ms_graph_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        start_datetime = f"{date}T{time}:00"
        start_dt_obj = datetime.strptime(start_datetime, "%Y-%m-%dT%H:%M:%S")
        end_dt_obj = start_dt_obj + timedelta(minutes=30)
        end_datetime = end_dt_obj.strftime("%Y-%m-%dT%H:%M:%S")

        event_data = {
            "subject": f"GoData Consultation: {user_name} ({company_name})",
            "start": {
                "dateTime": start_datetime,
                "timeZone": "SA Pacific Standard Time" 
            },
            "end": {
                "dateTime": end_datetime,
                "timeZone": "SA Pacific Standard Time"
            },
            "attendees": [
                {
                    "emailAddress": {"address": user_email, "name": user_name},
                    "type": "required"
                },
                {
                    "emailAddress": {"address": "juan.montiel@godata.com.ec", "name": "Juan Montiel"},
                    "type": "required"
                }
            ],
            "isOnlineMeeting": True,
            "onlineMeetingProvider": "teamsForBusiness"
        }
        
        endpoint = f"https://graph.microsoft.com/v1.0/users/{TARGET_CALENDAR}/events"
        response = requests.post(endpoint, headers=headers, json=event_data)
        
        if response.status_code == 201:
            event = response.json()
            teams_link = event.get("onlineMeeting", {}).get("joinUrl", "Link generated but hidden.")
            return f"Success! Meeting booked with Juan. Give the user this Teams link: {teams_link}. Also give conscise Details of the meeting: {event.get('subject')} on {date} at {time}."
        else:
            return f"Failed to book meeting. Error: {response.text}"
            
    except Exception as e:
        return f"System error booking meeting: {str(e)}"
    



@tool
def validate_email_format(email: str) -> str:
    """
    Validates an email address format.
    """
    # 1. Basic Regex Check
    regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    if not re.match(regex, email):
        return f"Error: '{email}' is not a valid email format."

    # 2. MX Record Check (Checks if the domain can actually receive mail)
    domain = email.split('@')[-1]
    try:
        dns.resolver.resolve(domain, 'MX')
        return f"Success: '{email}' appears to be a valid, reachable email address."
    except Exception:
        return f"Error: The domain '{domain}' does not seem to have a valid mail server."
    
@tool
def check_team_availability(date: str, time: str) -> str:
    """
    Checks if Juan is free at a specific date (YYYY-MM-DD) and time (HH:MM).
    Always call this before booking to avoid collisions.
    """
    try:
        token = get_ms_graph_token()
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        # Define the 30-minute window we want to check
        start_time = f"{date}T{time}:00"
        start_dt = datetime.strptime(start_time, "%Y-%m-%dT%H:%M:%S")
        end_dt = start_dt + timedelta(minutes=30)
        
        payload = {
            "schedules": ["juan.montiel@godata.com.ec"],
            "startTime": {"dateTime": start_time, "timeZone": "SA Pacific Standard Time"},
            "endTime": {"dateTime": end_dt.strftime("%Y-%m-%dT%H:%M:%S"), "timeZone": "SA Pacific Standard Time"},
            "availabilityViewInterval": 30
        }
        
        response = requests.post(
            "https://graph.microsoft.com/v1.0/users/juan.montiel@godata.com.ec/calendar/getSchedule",
            headers=headers,
            json=payload
        )
        
        data = response.json()
        for schedule in data.get('value', []):
            # If availabilityView is not '0' (Free), then they are busy
            if schedule.get('availabilityView', '0') != '0':
                return f"Conflict: {schedule.get('scheduleId')} is busy at that time. Suggest a different slot."
        
        return "Success: Juan is free at this time."
    except Exception as e:
        return f"Error checking availability: {str(e)}"