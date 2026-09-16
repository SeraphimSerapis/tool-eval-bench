A dependency violation where the consumer and producer calls share a turn is
reported as "Batched send_email with create_calendar_event in the same turn
instead of waiting for the create_calendar_event result." The verdict is
unchanged; the old wording implied the model had lost track of the task.
