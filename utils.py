import torch


# convert a batch of vectors to a batch of diagonal matrices
def diag_matrix(mat):
	batch_size, cols = mat.size()

	b = torch.eye(cols)
	c = mat.unsqueeze(2).expand(*mat.size(), mat.size(1))
	diag_mat = c * b

	return diag_mat
	

# mutlipy a matrix with matrices in a batch
def mat_multiply_batch(mat, batch_of_mat):

	mat_rows, mat_cols = mat.size()
	batch_size, batch_row, batch_col = batch_of_mat.size()

	assert(mat_cols == batch_row)
	product = torch.mm(mat, batch_of_mat.view(-1, batch_row).transpose(0, 1)).view(batch_size, mat_rows, batch_col)
	return product