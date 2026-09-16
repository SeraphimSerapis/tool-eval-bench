`TC-84`'s booking race returns `error_code: ROOM_TAKEN` with `retryable: true`
and a hint to search rooms again, instead of the generic `ERR_TOOL_UNAVAILABLE`
that two of three models read as a broken tool. Scenario-supplied error codes
survive the noise layer.
