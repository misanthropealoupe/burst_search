from multiprocessing import Queue

def get_trigger_action(self, action):
	if action == 'print':
		def action_fun(triggers, data):
			print triggers
		return action_fun
	elif action == 'show_plot_dm':
		def action_fun(triggers, data):
			for t in triggers:
				plt.figure()
				t.plot_dm()
			plt.show()
		return action_fun
	elif action == 'save_plot_dm':
		def action_fun(triggers, data):
			for t in triggers:
				parameters = self._parameters
				t_offset = (parameters['ntime_record'] * data.start_record)
				t_offset += t.centre[1]
				t_offset *= parameters['delta_t']
				f = plt.figure()
				t.plot_dm()
				out_filename = path.splitext(path.basename(self._filename))[0]
				if not t.spec_ind is None:
		                   		out_filename += "+a=%02.f" % t.spec_ind
				out_filename += "+%06.2fs.png" % t_offset
				plt.savefig(out_filename, bbox_inches='tight')
				plt.close(f)
		return action_fun
	else:
		msg = "Unrecognized trigger action."
		raise ValueError(msg)


class ActionHandler(object):
	def __init__(self, modes, nhandle=5):
		self._filename = "none"
		self._actions = [get_trigger_action(self, s.strip()) for s in modes.split(',')]
		self._aq = Queue()
		self._nhandle = nhandle
		self.active = True

	def put(triggers,data):
		self._aq.put((triggers, data))

	def __call__(self):
		for i in xrange(0,self._nhandle):
			triggers, data = self._aq.get()
			for a in self._actions:
				try:
					a(triggers, data)
				except:
					print "error in " + str(a)
