**A slowly answering server can no longer stall engine probing.** The probe timeout applied to
each read, so an endpoint that sent headers and then trickled its body kept detection waiting
indefinitely before a run could start. Each probe now ends after 5 s in total, and such an
overrun counts toward the two timeouts that end a probing session.
