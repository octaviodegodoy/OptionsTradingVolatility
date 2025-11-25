import datetime


def count_weekdays(start_date, days_to_add):
    weekdays_count = 0  # Counter for weekdays
    current_date = start_date  # Start counting from this date
    
    for _ in range(days_to_add):
        current_date += datetime.timedelta(days=1)  # Move to the next day
        if current_date.weekday() < 5:  # Check if it's a weekday (0-4 are weekdays)
            weekdays_count += 1  # Increment weekday counter

    return weekdays_count