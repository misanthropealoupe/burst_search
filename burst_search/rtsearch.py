import search
class Searcher(object):
	pass
class RtSearch(Searcher):
	def __init__(self, data_source, ring_buffer, action_hander):
		self._data_source = data_source
		self._ring_buffer = ring_buffer
		self._chunk_size = data_source.chunk_size

	def __call__(self):
		p = self._data_source.queue.get()
		rb_data = ring_buffer(p)
		triggers = search.basic(rb_data)
		del p
		return triggers
