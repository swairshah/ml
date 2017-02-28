import numpy as np
import matplotlib.pyplot as plt

def run(important_sampling, weightedXY):
	mean = [0, 0]
	cov = [[1, 0.5], [0.5, 1]]

	m = 5000
	# np.random.seed(2017)
	x, y = np.random.multivariate_normal(mean, cov, m).T
	# base_fit is a function which takes in x and returns an estimate for y
	fit = np.polyfit(x, y, 1, full = True)
	base_fit = np.poly1d(fit[0])
	base_predict_y = base_fit(x)
	base_error = fit[1][0]
	# print "use all the data points, base error = " + str(base_error)

	# plt.plot(x, y, 'x', x, base_predict_y, 'black', linewidth = 3.0)
	# plt.show()

	# initial random selection 
	m_samples = 50 
	index = [i for i in range(m)]
	sample_id = np.random.choice(index, m_samples)

	fit = np.polyfit(x[sample_id], y[sample_id], 1) 
	fit_fn = np.poly1d(fit) 
	predict_y = fit_fn(x)
	w = (predict_y - y) ** 2
	initial_error = sum(w)
	w /= initial_error

	min_error = initial_error
	best_fit = np.copy(fit)
	best_samples = np.copy(sample_id)
	# print "initial pass, initial error = " + str(initial_error)

	# plt.title('initial.'+ ', base_err= ' + str(base_error) \
	#  			+ ', initial_err=' + str(initial_error))
	# plt.plot(x, y, 'x', x[sample_id], y[sample_id], 'ro', x, predict_y, 'g--')
	# plt.plot(x, base_predict_y, 'black', linewidth = 3.0)
	# plt.show()

	# iterations 
	iters = 10
	update = 0
        sample_errors = []
	for it in range(iters):
		# use important sampling or simple random sampling
		if important_sampling:
			sample_id = np.random.choice(index, m_samples, p = w)
		else:
			sample_id = np.random.choice(index, m_samples)
		# scaleXY or not
		if weightedXY:
			# sqrt_w = np.sqrt(w[sample_id])
			# fit = np.polyfit(x[sample_id] / sqrt_w, y[sample_id] / sqrt_w, 1) 
			fit = np.polyfit(x[sample_id] / w[sample_id], y[sample_id] / w[sample_id], 1) 
		else:
			fit = np.polyfit(x[sample_id], y[sample_id], 1) 
		fit_fn = np.poly1d(fit) 
		predict_y = fit_fn(x)
		e = (predict_y - y) ** 2
		cur_error = sum(e)

                sample_errors.append(cur_error)
		# print "iter=" + str(it) + " cur_error = " + str(cur_error)
		if cur_error < min_error:
			# print "min_error is updated"
			update += 1
			min_error = cur_error 
			if important_sampling:
				w = np.copy(e)
				w /= cur_error 
			best_fit = np.copy(fit)
			best_samples = np.copy(sample_id)
		# plt.title('iteration=' + str(it) + ', base_err= ' + str(base_error) \
		#  			+ ', cur_err=' + str(cur_error))
		# plt.plot(x, y, 'x', x[sample_id], y[sample_id], 'ro', x, predict_y, 'g--')
		# plt.plot(x, base_predict_y, 'black', linewidth = 3.0)
		# plt.show()

	# plot the best model 
	# best_fit_fn = np.poly1d(best_fit) 
	# plt.title('end,' + ' base_err= ' + str(base_error) \
	#  			+ ', min_err=' + str(min_error))
	# plt.plot(x, y, 'x', x[best_samples], y[best_samples], 'ro', x, best_fit_fn(x), 'g--')
	# plt.plot(x, base_predict_y, 'black', linewidth = 3.0)
	# plt.show()
	
        #return [update, update == 0, (min_error - base_error) / base_error]
        return sample_errors

repeat = 1
important_sampling = True
weightedXY = True
update, initial_is_best, error_increase = 0, 0, 0
"""for i in range(repeat):
	res = run(important_sampling, weightedXY)
	update += res[0] 
	error_increase += res[2]
	if res[1]:
		initial_is_best += 1

print 'important sampling= ' + str(important_sampling)
print 'weightedXY= ' + str(weightedXY)
print 'average updates= ' + str(1.0 * update / repeat)
print 'initial random selection gives best model= %f%%' % (100.0 * initial_is_best / repeat)
print 'average error increased compared to the base model= %f%%' % (100.0 * error_increase / repeat)
"""

imp = run(important_sampling, True)
uni = run(important_sampling, False)
plt.plot(imp)
plt.plot(uni, c = "orange")
plt.legend(["importance","uniform"])
plt.show()

